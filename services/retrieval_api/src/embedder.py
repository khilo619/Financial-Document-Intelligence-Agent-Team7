import os

import torch
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    """
    Handles text embeddings using LangChain's
    Hugging Face integration.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ):
        # Use local model path from environment if provided.
        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "BAAI/bge-large-en-v1.5",
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Embedding model: {model_name}")
        print(f"Embedding device: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": self.device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    def encode(self, text: str) -> list[float]:
        """
        Convert text into a normalized dense embedding.
        """

        embedding = self.embeddings.embed_query(text)

        return embedding
