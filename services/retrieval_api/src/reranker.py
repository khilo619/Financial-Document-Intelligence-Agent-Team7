import os

import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class Reranker:
    """
    Reranks retrieved documents using LangChain's
    Hugging Face Cross-Encoder integration.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ):
        # Use local model path from environment if provided.
        model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME",
            "BAAI/bge-reranker-large",
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Reranker model: {model_name}")
        print(f"Reranker device: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.model = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs={
                "device": self.device,
            },
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        if not results:
            return []

        pairs = [(query, result.get("content", "")) for result in results]

        scores = self.model.score(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            updated_result = result.copy()

            updated_result["rerank_score"] = float(score)

            reranked_results.append(updated_result)

        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        return reranked_results[:top_n]
