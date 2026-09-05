from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    """
    Handles text embeddings using LangChain's
    Hugging Face integration.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
    ):
        # Initialize the Hugging Face embedding model
        # through LangChain.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
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
