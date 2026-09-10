"""
services.eval_service.src.langfuse_reporter: MLOps and observability bridge for Project LEDGER.
Manages:
- Langfuse client initialization with environment variables (.env)
- Dataset syncing (registering benchmark questions as versioned Langfuse datasets)
- Quantitative score logging (Exact Match, Token F1, Numerical Accuracy, Retrieval Recall)
- Graceful offline fallback when Langfuse server is unavailable or disabled
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("LangfuseReporter")

try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse package not installed. Operating in offline logging mode.")


class LangfuseReporter:
    """
    MLOps reporter managing telemetry, datasets, and score dispatches to Langfuse.
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        enabled: bool | None = None,
    ):
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY", "")
        self.host = host or os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")

        if enabled is not None:
            self.enabled = enabled
        else:
            tracing_env = os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower()
            self.enabled = tracing_env in ("true", "1", "yes")

        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initializes Langfuse client if enabled and keys are provided."""
        if not self.enabled:
            logger.info("Langfuse tracing is disabled via configuration (offline mode).")
            return

        if not LANGFUSE_AVAILABLE:
            self.enabled = False
            return

        if not self.public_key or not self.secret_key:
            logger.warning("Langfuse public/secret keys missing. Falling back to local offline mode.")
            self.enabled = False
            return

        # Check if keys are placeholders from .env.example
        if "sample" in self.public_key or "sample" in self.secret_key:
            logger.info("Sample/dummy Langfuse keys detected. Operating in simulated offline mode.")
            self.enabled = False
            return

        try:
            self.client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            logger.info("Langfuse client connected to host: %s", self.host)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize Langfuse client: %s. Using offline mode.", exc)
            self.enabled = False
            self.client = None

    def sync_dataset(self, dataset_name: str, questions: list[dict[str, Any]]) -> int:
        """
        Syncs benchmark items to a Langfuse Dataset.
        Returns the number of synced items.
        """
        if not self.enabled or not self.client:
            logger.info(
                "[OFFLINE] Registered %d questions to local dataset '%s'.",
                len(questions),
                dataset_name,
            )
            return len(questions)

        try:
            # Create or ensure dataset exists
            try:
                self.client.create_dataset(
                    name=dataset_name,
                    description="Project LEDGER Practice Benchmark Gold Dataset (Set A)",
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dataset creation check for '%s': %s", dataset_name, exc)

            synced_count = 0
            for q in questions:
                q_id = q.get("question_id", "unknown")
                q_text = q.get("question_text", "")
                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    input={"question_id": q_id, "question_text": q_text},
                    expected_output={
                        "ground_truth_answer": q.get("ground_truth_answer"),
                        "answer_type": q.get("answer_type"),
                        "scale": q.get("scale"),
                    },
                    metadata={
                        "task_family": q.get("task_family"),
                        "difficulty_tier": q.get("difficulty_tier"),
                    },
                )
                synced_count += 1

            self.client.flush()
            logger.info("Synced %d items to Langfuse dataset '%s'.", synced_count, dataset_name)
            return synced_count

        except Exception as exc:  # noqa: BLE001
            logger.error("Error syncing dataset to Langfuse: %s", exc)
            return len(questions)

    def log_evaluation_result(
        self,
        question_id: str,
        question_text: str,
        prediction_answer: Any,
        ground_truth_answer: Any,
        scores: dict[str, float],
        metadata: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> str:
        """
        Logs an evaluation trace and dispatches quantitative scores to Langfuse.
        Returns a trace ID or mock ID.
        """
        meta = metadata or {}
        meta["latency_ms"] = round(latency_ms, 2)

        if not self.enabled or not self.client:
            logger.debug(
                "[OFFLINE TRACE] Q: %s | ExactMatch: %.1f | NumAcc: %.1f | F1: %.2f",
                question_id,
                scores.get("exact_match", 0.0),
                scores.get("numerical_accuracy", 0.0),
                scores.get("f1_score", 0.0),
            )
            return f"mock_trace_{question_id}"

        try:
            if hasattr(self.client, "trace"):
                trace = self.client.trace(
                    name="ledger_eval_item",
                    session_id=meta.get("session_id", "eval_practice_run"),
                    input={"question_id": question_id, "question": question_text},
                    output={"prediction": str(prediction_answer)},
                    metadata=meta,
                    tags=[
                        meta.get("task_family", "general"),
                        f"diff_tier_{meta.get('difficulty_tier', 1)}",
                    ],
                )
                trace_id = trace.id
                for score_name, score_val in scores.items():
                    trace.score(
                        name=score_name,
                        value=float(score_val),
                        comment=f"Ground truth: {ground_truth_answer}",
                    )
                return trace_id
            elif hasattr(self.client, "create_score"):
                trace_id = (
                    self.client.create_trace_id() if hasattr(self.client, "create_trace_id") else f"trace_{question_id}"
                )
                for score_name, score_val in scores.items():
                    self.client.create_score(
                        name=score_name,
                        value=float(score_val),
                        trace_id=trace_id,
                        comment=f"Ground truth: {ground_truth_answer}",
                    )
                return trace_id
            else:
                return f"trace_{question_id}"

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to dispatch scores to Langfuse for %s: %s", question_id, exc)
            return f"err_trace_{question_id}"

    def flush(self) -> None:
        """Flushes buffered events to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Error during Langfuse client flush: %s", exc)
