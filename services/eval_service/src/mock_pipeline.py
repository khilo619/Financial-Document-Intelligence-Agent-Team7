"""
services.eval_service.src.mock_pipeline: Decoupled pipeline clients for evaluation.
Provides:
- PipelineClient (Protocol interface)
- MockPipelineClient (Standalone emulator with configurable error injection for Day 2/3)
- HttpPipelineClient (Production client pointing to orchestrator-api or agent-service)
"""

import logging
import random
import time
from typing import Any, Protocol

import httpx

from shared.config import ServiceName, get_service_url
from shared.models import AnswerType, Citation, StrictAnswer

logger = logging.getLogger("EvalPipelineClient")


class PipelineClient(Protocol):
    """Protocol interface defining how evaluation interacts with any RAG pipeline."""

    def query(
        self,
        question_id: str,
        question_text: str,
        gold_item: dict[str, Any] | None = None,
    ) -> tuple[StrictAnswer, list[dict[str, Any]], float]:
        """
        Executes a question through the pipeline.
        Returns:
            tuple of (StrictAnswer, retrieved_chunks, latency_ms)
        """
        ...


class MockPipelineClient:
    """
    Standalone pipeline simulator.
    Generates realistic, schema-compliant StrictAnswer objects based on gold benchmark items.
    Allows testing evaluation metrics, Langfuse experiment logging, and failure analysis
    without requiring upstream microservices or external API keys.
    """

    def __init__(
        self,
        error_rate: float = 0.15,
        latency_range_ms: tuple[float, float] = (45.0, 180.0),
        seed: int = 42,
    ):
        self.error_rate = error_rate
        self.latency_range_ms = latency_range_ms
        self.rng = random.Random(seed)

    def query(
        self,
        question_id: str,
        question_text: str,
        gold_item: dict[str, Any] | None = None,
    ) -> tuple[StrictAnswer, list[dict[str, Any]], float]:
        start_time = time.perf_counter()
        simulated_latency = self.rng.uniform(*self.latency_range_ms)
        # Sleep for a tiny fraction of simulated latency to keep tests fast
        time.sleep(min(simulated_latency / 1000.0, 0.01))

        if not gold_item:
            # Fallback when no gold item is provided
            answer = StrictAnswer(
                answer_type=AnswerType.INSUFFICIENT_EVIDENCE.value,
                evidence=[],
                params={"reason": "No context or gold item provided to mock."},
            )
            return answer, [], simulated_latency

        gold_type = gold_item.get("answer_type", "span")
        gold_answer = gold_item.get("ground_truth_answer")
        gold_ev_list = gold_item.get("gold_evidence", [])
        derivation = gold_item.get("derivation") or "calculated_formula"

        # Build realistic citations from gold evidence
        citations: list[Citation] = []
        retrieved_chunks: list[dict[str, Any]] = []

        for ev in gold_ev_list:
            doc_id = ev.get("source_document") or "doc_sample.pdf"
            page = ev.get("source_page", 1) or 1
            citations.append(
                Citation(
                    document_id=doc_id,
                    page=page,
                    section="Financial Disclosures",
                    bbox=[50.0, 100.0, 450.0, 200.0],
                )
            )
            retrieved_chunks.append({"document_id": doc_id, "page": page, "score": 0.89})

        # Inject controlled errors to test failure modes
        should_inject_error = self.rng.random() < self.error_rate

        if should_inject_error:
            error_type = self.rng.choice(["calc_error", "retrieval_miss", "abstention"])

            if error_type == "calc_error" and isinstance(gold_answer, (int, float)):
                # Simulate calculation mistake: valid math formula that yields wrong financial figure
                offset = 10.0
                corrupted_val = round(float(gold_answer) + offset, 2)
                try:
                    corrupted_formula = (
                        f"({derivation}) + {offset}"
                        if derivation and derivation != "calculated_formula"
                        else str(corrupted_val)
                    )
                    answer = StrictAnswer(
                        answer_type=AnswerType.CALCULATED.value,
                        evidence=citations or [Citation(document_id="doc_001.pdf", page=1)],
                        params={
                            "value": corrupted_val,
                            "formula": corrupted_formula,
                        },
                    )
                except Exception:
                    answer = StrictAnswer(
                        answer_type=AnswerType.CALCULATED.value,
                        evidence=citations or [Citation(document_id="doc_001.pdf", page=1)],
                        params={
                            "value": corrupted_val,
                            "formula": str(corrupted_val),
                        },
                    )
                return answer, retrieved_chunks, simulated_latency

            elif error_type == "retrieval_miss":
                # Replace evidence with distractor
                distractor_chunks = [{"document_id": "distractor_unrelated_2018.pdf", "page": 4, "score": 0.42}]
                distractor_citations = [Citation(document_id="distractor_unrelated_2018.pdf", page=4)]
                answer = StrictAnswer(
                    answer_type=AnswerType.DIRECT.value,
                    evidence=distractor_citations,
                    params={"value": "Distractor factual extract"},
                )
                return answer, distractor_chunks, simulated_latency

            elif error_type == "abstention":
                answer = StrictAnswer(
                    answer_type=AnswerType.INSUFFICIENT_EVIDENCE.value,
                    evidence=[],
                    params={"reason": "Grounding facts were missing from retrieved pages."},
                )
                return answer, retrieved_chunks, simulated_latency

        # Generate correct ground-truth response
        if gold_type == "arithmetic":
            numeric_val = float(gold_answer) if gold_answer is not None else 0.0
            try:
                answer = StrictAnswer(
                    answer_type=AnswerType.CALCULATED.value,
                    evidence=citations or [Citation(document_id="doc_default.pdf", page=1)],
                    params={"value": numeric_val, "formula": derivation},
                )
            except Exception:
                answer = StrictAnswer(
                    answer_type=AnswerType.CALCULATED.value,
                    evidence=citations or [Citation(document_id="doc_default.pdf", page=1)],
                    params={"value": numeric_val, "formula": str(numeric_val)},
                )
        elif gold_type == "multi-span":
            vals = (
                [str(x) for x in gold_answer] if isinstance(gold_answer, list) else [str(gold_answer), "secondary_item"]
            )
            if len(vals) < 2:
                vals.append("supplemental_span")
            answer = StrictAnswer(
                answer_type=AnswerType.MULTI_SPAN.value,
                evidence=citations or [Citation(document_id="doc_default.pdf", page=1)],
                params={"values": vals},
            )
        elif gold_type == "unanswerable":
            answer = StrictAnswer(
                answer_type=AnswerType.INSUFFICIENT_EVIDENCE.value,
                evidence=[],
                params={"reason": "Benchmark item is unanswerable from corpus."},
            )
        else:
            # Default to direct answer
            val = str(gold_answer) if gold_answer is not None else "Factual direct answer"
            answer = StrictAnswer(
                answer_type=AnswerType.DIRECT.value,
                evidence=citations or [Citation(document_id="doc_default.pdf", page=1)],
                params={"value": val},
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0 + simulated_latency
        return answer, retrieved_chunks, latency_ms


class HttpPipelineClient:
    """
    Live HTTP pipeline client.
    Connects to orchestrator-api (Port 8001) or agent-service (Port 8004) via HTTP POST /ask.
    """

    def __init__(self, endpoint_url: str | None = None, timeout: float = 90.0):
        self.endpoint_url = endpoint_url or f"{get_service_url(ServiceName.ORCHESTRATOR.value)}/ask"
        self.timeout = timeout

    def query(
        self,
        question_id: str,
        question_text: str,
        gold_item: dict[str, Any] | None = None,
    ) -> tuple[StrictAnswer, list[dict[str, Any]], float]:
        start = time.perf_counter()

        payload = {
            "query": question_text,
            "session_id": f"eval_{question_id}",
            "include_debug_traces": True,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.endpoint_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.perf_counter() - start) * 1000.0
            answer_payload = data.get("answer", {})
            strict_answer = StrictAnswer(**answer_payload)
            retrieved = data.get("retrieved_chunks") or [
                {"document_id": c.document_id, "page": c.page} for c in strict_answer.evidence
            ]
            return strict_answer, retrieved, latency_ms

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error("HttpPipelineClient failed for question %s: %s", question_id, exc)
            fallback = StrictAnswer(
                answer_type=AnswerType.INSUFFICIENT_EVIDENCE.value,
                evidence=[],
                params={"reason": f"Pipeline request failed: {exc}"},
            )
            return fallback, [], latency_ms
