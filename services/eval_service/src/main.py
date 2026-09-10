"""
services.eval_service.src.main: FastAPI server for Project LEDGER evaluation & MLOps suite.
Owned by Khaled (Member 5 - Evaluation & MLOps Lead).
Exposes endpoints for running automated benchmarks, retrieving metrics, and generating failure analyses.
"""

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.config import ServiceName, get_service_url

from .benchmark import load_benchmark_dataset, run_benchmark
from .langfuse_reporter import LangfuseReporter
from .mock_pipeline import (
    HttpPipelineClient,
    MockPipelineClient,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("EvalService")

app = FastAPI(
    title="Project LEDGER - Evaluation & MLOps Service",
    description="Automated benchmarking on questions_setA_practice.json, quantitative metrics computation, and Langfuse tracing.",
    version="1.0.0",
)

# In-memory store for the latest benchmark report
LATEST_REPORT: dict[str, Any] | None = None


class BenchmarkRunRequest(BaseModel):
    sample_size: int = Field(default=10, ge=1, le=500, description="Number of questions to run")
    task_family: str | None = Field(default=None, description="Optional task family filter")
    use_mock: bool = Field(
        default_factory=lambda: os.getenv("EVAL_USE_MOCK", "false").lower() in ("true", "1", "yes"),
        description="Whether to use MockPipelineClient or live HTTP (default false: evaluates live agent)",
    )
    endpoint_url: str | None = Field(default=None, description="Custom HTTP endpoint for live pipeline")


class BenchmarkRunResponse(BaseModel):
    total_evaluated: int
    exact_match: float
    f1_score: float
    numerical_accuracy: float
    retrieval_recall: float
    avg_latency_ms: float
    pass_rate: float
    failures_count: int
    task_family_breakdown: dict[str, Any]
    status: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": ServiceName.EVAL.value,
        "port": 8006,
    }


@app.post("/run_benchmark", response_model=BenchmarkRunResponse)
def execute_benchmark(request: BenchmarkRunRequest):
    """
    Executes automated benchmark evaluation sweep across questions_setA_practice.json.
    Computes Exact Match, F1, Numerical Accuracy (epsilon=0.01), and Retrieval Recall@K.
    """
    global LATEST_REPORT
    logger.info(
        "Executing benchmark run (sample_size=%d, mock=%s, task_family=%s)",
        request.sample_size,
        request.use_mock,
        request.task_family,
    )

    try:
        dataset = load_benchmark_dataset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark dataset: {exc}") from exc

    default_orchestrator_url = f"{get_service_url(ServiceName.ORCHESTRATOR.value)}/ask"
    client = (
        MockPipelineClient()
        if request.use_mock
        else HttpPipelineClient(endpoint_url=request.endpoint_url or default_orchestrator_url)
    )
    reporter = LangfuseReporter()

    report = run_benchmark(
        questions=dataset,
        client=client,
        reporter=reporter,
        sample_size=request.sample_size,
        task_family=request.task_family,
    )

    LATEST_REPORT = {
        "total_evaluated": report.total_evaluated,
        "exact_match": report.mean_exact_match,
        "f1_score": report.mean_f1_score,
        "numerical_accuracy": report.mean_numerical_accuracy,
        "retrieval_recall": report.mean_retrieval_recall,
        "avg_latency_ms": report.mean_latency_ms,
        "pass_rate": report.pass_rate,
        "failures_count": len(report.failures),
        "task_family_breakdown": report.task_family_breakdown,
        "failures": report.failures,
        "status": "completed",
    }

    return BenchmarkRunResponse(
        total_evaluated=report.total_evaluated,
        exact_match=report.mean_exact_match,
        f1_score=report.mean_f1_score,
        numerical_accuracy=report.mean_numerical_accuracy,
        retrieval_recall=report.mean_retrieval_recall,
        avg_latency_ms=report.mean_latency_ms,
        pass_rate=report.pass_rate,
        failures_count=len(report.failures),
        task_family_breakdown=report.task_family_breakdown,
        status="completed",
    )


@app.get("/metrics")
def get_metrics():
    """Returns the latest evaluation summary and benchmark targets."""
    return {
        "benchmark_file": "questions_setA_practice.json",
        "baseline_targets": {
            "exact_match_target": 0.75,
            "numerical_accuracy_target": 0.85,
            "recall_at_5_target": 0.80,
            "epsilon_relative_tolerance": 0.01,
        },
        "latest_run": LATEST_REPORT,
    }


@app.get("/failure_analysis")
def get_failure_analysis(top_n: int = 5):
    """
    Returns the failure analysis formatted according to the required 5-Example Failure Analysis standard.
    """
    if not LATEST_REPORT or not LATEST_REPORT.get("failures"):
        return {
            "status": "no_failures_recorded",
            "message": "Run /run_benchmark to generate failure cases.",
            "examples": [],
        }

    failures = LATEST_REPORT["failures"][:top_n]
    formatted_examples = []
    for idx, f in enumerate(failures, 1):
        formatted_examples.append(
            {
                "example_index": idx,
                "question_id": f["question_id"],
                "question_text": f["question_text"],
                "expected": f["ground_truth_answer"],
                "predicted": f["predicted_answer"],
                "root_cause_stage": f["failure_stage"],
                "diagnosis": f["diagnosis"],
                "remediation": f"Remediation plan targeted for {f['failure_stage']} layer.",
            }
        )

    return {
        "total_failures_recorded": len(LATEST_REPORT["failures"]),
        "presented_failures_count": len(formatted_examples),
        "examples": formatted_examples,
    }
