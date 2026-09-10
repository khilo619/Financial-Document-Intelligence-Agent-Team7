"""
services.eval_service.src.benchmark: Automated benchmark execution harness for Project LEDGER.
Iterates over questions_setA_practice.json, executes pipeline queries, computes
Exact Match, F1, Numerical Accuracy (epsilon=0.01), Retrieval Recall@K, and aggregates reports.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shared.models import StrictAnswer

from .langfuse_reporter import LangfuseReporter
from .metrics import (
    exact_match,
    numerical_accuracy,
    retrieval_recall_at_k,
    token_f1,
)
from .mock_pipeline import (
    HttpPipelineClient,
    MockPipelineClient,
    PipelineClient,
)

logger = logging.getLogger("BenchmarkRunner")


@dataclass
class QuestionEvaluationResult:
    question_id: str
    question_text: str
    task_family: str
    difficulty_tier: int
    ground_truth_answer: Any
    predicted_answer: Any
    answer_type: str
    exact_match: float
    f1_score: float
    numerical_accuracy: float
    retrieval_recall: float
    latency_ms: float
    passed: bool
    failure_stage: str | None = None
    diagnosis: str | None = None


@dataclass
class BenchmarkReport:
    total_evaluated: int
    mean_exact_match: float
    mean_f1_score: float
    mean_numerical_accuracy: float
    mean_retrieval_recall: float
    mean_latency_ms: float
    pass_rate: float
    task_family_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)


def load_benchmark_dataset(file_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Loads benchmark questions from JSON file."""
    if file_path is None:
        # Default search path in workspace root
        current_dir = Path(__file__).resolve().parent
        candidates = [
            current_dir / ".." / ".." / ".." / "questions_setA_practice.json",
            Path("questions_setA_practice.json"),
        ]
        target = None
        for c in candidates:
            if c.resolve().exists():
                target = c.resolve()
                break
        if not target:
            raise FileNotFoundError("questions_setA_practice.json not found in repository root.")
        file_path = target

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d questions from %s", len(data), file_path)
    return data


def extract_predicted_value(answer: StrictAnswer) -> Any:
    """Extracts scalar or text value from StrictAnswer params."""
    params = answer.params or {}
    if "value" in params:
        return params["value"]
    if "values" in params:
        return params["values"]
    if "reason" in params:
        return params["reason"]
    return None


def run_benchmark(
    questions: list[dict[str, Any]],
    client: PipelineClient | None = None,
    reporter: LangfuseReporter | None = None,
    sample_size: int | None = None,
    task_family: str | None = None,
) -> BenchmarkReport:
    """
    Executes benchmark evaluation loop across questions.
    """
    if client is None:
        client = MockPipelineClient()
    if reporter is None:
        reporter = LangfuseReporter()

    # Filter dataset
    filtered = questions
    if task_family:
        filtered = [q for q in filtered if q.get("task_family") == task_family]

    if sample_size and sample_size > 0:
        filtered = filtered[:sample_size]

    total = len(filtered)
    if total == 0:
        logger.warning("No questions matched benchmark filter criteria.")
        return BenchmarkReport(
            total_evaluated=0,
            mean_exact_match=0.0,
            mean_f1_score=0.0,
            mean_numerical_accuracy=0.0,
            mean_retrieval_recall=0.0,
            mean_latency_ms=0.0,
            pass_rate=0.0,
        )

    logger.info("Starting benchmark run over %d questions...", total)

    results: list[QuestionEvaluationResult] = []
    task_metrics: dict[str, list[dict[str, float]]] = {}

    for idx, q in enumerate(filtered, 1):
        q_id = q.get("question_id", f"Q_{idx}")
        q_text = q.get("question_text", "")
        gold_ans = q.get("ground_truth_answer")
        gold_type = q.get("answer_type", "span")
        gold_scale = q.get("scale")
        gold_ev = q.get("gold_evidence", [])
        tf = q.get("task_family", "general")
        tier = q.get("difficulty_tier", 1)

        # Execute through pipeline
        strict_answer, retrieved_chunks, latency_ms = client.query(
            question_id=q_id,
            question_text=q_text,
            gold_item=q,
        )

        pred_val = extract_predicted_value(strict_answer)

        # Compute core metrics
        em = exact_match(pred_val, gold_ans)
        f1 = token_f1(pred_val, gold_ans)

        # Determine numerical accuracy if applicable
        if gold_type in ("arithmetic", "count") or isinstance(gold_ans, (int, float)):
            num_acc = numerical_accuracy(
                pred_val=pred_val,
                true_val=gold_ans,
                pred_scale=None,
                true_scale=gold_scale,
                epsilon=0.01,
            )
            # A question is passed if numerical accuracy meets epsilon tolerance
            passed = num_acc == 1.0
        elif gold_type == "unanswerable":
            passed = strict_answer.answer_type == "insufficient_evidence"
            num_acc = 1.0 if passed else 0.0
        else:
            # Span or text question passes on exact match or strong F1 (> 0.8)
            num_acc = em
            passed = em == 1.0 or f1 >= 0.8

        # Compute retrieval recall
        ret_recall = retrieval_recall_at_k(retrieved_chunks, gold_ev)

        # Failure diagnosis
        failure_stage = None
        diagnosis = None
        if not passed:
            if ret_recall < 0.5:
                failure_stage = "Retrieval"
                diagnosis = (
                    f"Relevant gold evidence documents {[g.get('source_document') for g in gold_ev]} "
                    f"were not found in top retrieved chunks."
                )
            elif strict_answer.answer_type == "calculated" and num_acc == 0.0:
                failure_stage = "Reasoning & Calculation"
                diagnosis = (
                    f"Formula calculation error: predicted {pred_val} but expected {gold_ans} "
                    f"(scale={gold_scale}, derivation={q.get('derivation')})."
                )
            elif strict_answer.answer_type == "insufficient_evidence" and q.get("is_answerable", True):
                failure_stage = "Agent Abstention"
                diagnosis = "Agent incorrectly abstained on an answerable question."
            else:
                failure_stage = "Extraction / Schema"
                diagnosis = f"Extracted value '{pred_val}' did not match gold answer '{gold_ans}'."

        res = QuestionEvaluationResult(
            question_id=q_id,
            question_text=q_text,
            task_family=tf,
            difficulty_tier=tier,
            ground_truth_answer=gold_ans,
            predicted_answer=pred_val,
            answer_type=strict_answer.answer_type,
            exact_match=em,
            f1_score=f1,
            numerical_accuracy=num_acc,
            retrieval_recall=ret_recall,
            latency_ms=latency_ms,
            passed=passed,
            failure_stage=failure_stage,
            diagnosis=diagnosis,
        )
        results.append(res)

        # Log to Langfuse
        scores_dict = {
            "exact_match": em,
            "f1_score": f1,
            "numerical_accuracy": num_acc,
            "retrieval_recall": ret_recall,
        }
        reporter.log_evaluation_result(
            question_id=q_id,
            question_text=q_text,
            prediction_answer=pred_val,
            ground_truth_answer=gold_ans,
            scores=scores_dict,
            metadata={"task_family": tf, "difficulty_tier": tier},
            latency_ms=latency_ms,
        )

        # Track per-task metrics
        if tf not in task_metrics:
            task_metrics[tf] = []
        task_metrics[tf].append(scores_dict)

    reporter.flush()

    # Aggregate summaries
    mean_em = sum(r.exact_match for r in results) / total
    mean_f1 = sum(r.f1_score for r in results) / total
    mean_num_acc = sum(r.numerical_accuracy for r in results) / total
    mean_ret_recall = sum(r.retrieval_recall for r in results) / total
    mean_latency = sum(r.latency_ms for r in results) / total
    passed_count = sum(1 for r in results if r.passed)
    pass_rate = passed_count / total

    # Task breakdown
    breakdown: dict[str, dict[str, float]] = {}
    for tf, scores_list in task_metrics.items():
        n = len(scores_list)
        breakdown[tf] = {
            "count": n,
            "exact_match": round(sum(s["exact_match"] for s in scores_list) / n, 4),
            "f1_score": round(sum(s["f1_score"] for s in scores_list) / n, 4),
            "numerical_accuracy": round(sum(s["numerical_accuracy"] for s in scores_list) / n, 4),
            "retrieval_recall": round(sum(s["retrieval_recall"] for s in scores_list) / n, 4),
        }

    failures_list = [asdict(r) for r in results if not r.passed]

    return BenchmarkReport(
        total_evaluated=total,
        mean_exact_match=round(mean_em, 4),
        mean_f1_score=round(mean_f1, 4),
        mean_numerical_accuracy=round(mean_num_acc, 4),
        mean_retrieval_recall=round(mean_ret_recall, 4),
        mean_latency_ms=round(mean_latency, 2),
        pass_rate=round(pass_rate, 4),
        task_family_breakdown=breakdown,
        failures=failures_list,
        results=[asdict(r) for r in results],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Project LEDGER automated benchmark suite.")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of questions to evaluate")
    parser.add_argument("--task-family", type=str, default=None, help="Filter by task_family")
    parser.add_argument(
        "--mock", dest="mock", action="store_true", default=True, help="Use MockPipelineClient (default)"
    )
    parser.add_argument("--live", dest="mock", action="store_false", help="Use live HTTP pipeline (orchestrator-api)")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8001/ask",
        help="HTTP endpoint when not using mock",
    )

    args = parser.parse_args()

    dataset = load_benchmark_dataset()

    client = MockPipelineClient() if args.mock else HttpPipelineClient(endpoint_url=args.endpoint)
    report = run_benchmark(
        questions=dataset,
        client=client,
        sample_size=args.sample_size,
        task_family=args.task_family,
    )

    print("\n=======================================================")
    print("           PROJECT LEDGER BENCHMARK REPORT             ")
    print("=======================================================")
    print(f"Total Evaluated:        {report.total_evaluated}")
    print(f"Overall Pass Rate:      {report.pass_rate * 100:.1f}%")
    print(f"Mean Exact Match:       {report.mean_exact_match:.4f}")
    print(f"Mean Token F1:          {report.mean_f1_score:.4f}")
    print(f"Mean Numerical Acc:     {report.mean_numerical_accuracy:.4f} (epsilon=0.01)")
    print(f"Mean Retrieval Recall:  {report.mean_retrieval_recall:.4f}")
    print(f"Average Latency:        {report.mean_latency_ms:.1f} ms")
    print("=======================================================\n")
    if report.failures:
        print(f"Sample Failure Analysis ({len(report.failures)} failures recorded):")
        for f in report.failures[:3]:
            print(f"  - [{f['question_id']}] Stage: {f['failure_stage']} | {f['diagnosis']}")
