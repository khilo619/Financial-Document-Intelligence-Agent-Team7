"""
eval_service: Benchmark evaluation engine and Langfuse observability.
Owned by Khaled (Member 5 - Evaluation & MLOps Lead).
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from shared.config import ServiceName

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("EvalService")

app = FastAPI(
    title="Project LEDGER - Evaluation & MLOps Service",
    description="Automated benchmarking on questions_setA_practice.json, metrics calculation, and Langfuse tracing.",
    version="0.1.0",
)


class BenchmarkRunRequest(BaseModel):
    sample_size: int = 5
    task_family: str | None = None


class BenchmarkRunResponse(BaseModel):
    total_evaluated: int
    exact_match: float
    f1_score: float
    numerical_accuracy: float
    avg_latency_ms: float
    status: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": ServiceName.EVAL.value,
        "port": 8006,
    }


@app.post("/run_benchmark", response_model=BenchmarkRunResponse)
def run_benchmark(request: BenchmarkRunRequest):
    """
    Runs automated evaluation loop across benchmark questions and computes metrics.
    Full implementation will be completed in Khaled's feature branch.
    """
    logger.info("Triggered benchmark evaluation run (sample_size=%d)", request.sample_size)

    # Scaffold baseline metrics
    return BenchmarkRunResponse(
        total_evaluated=request.sample_size,
        exact_match=1.0,
        f1_score=1.0,
        numerical_accuracy=1.0,
        avg_latency_ms=120.5,
        status="completed",
    )


@app.get("/metrics")
def get_metrics():
    """Returns the latest evaluation report."""
    return {
        "benchmark_file": "questions_setA_practice.json",
        "baseline_targets": {
            "exact_match_target": 0.75,
            "numerical_accuracy_target": 0.85,
            "recall_at_5_target": 0.80,
        },
    }
