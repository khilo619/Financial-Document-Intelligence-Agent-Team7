from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around the BGE embedding model.

    This class is responsible for converting text into dense
    vector representations that can be stored and searched in Qdrant.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        # Load the pre-trained BGE embedding model.
        # The model converts text into high-dimensional dense vectors.
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        """
        Convert a text string into a normalized embedding vector.

        Args:
            text: The text to convert into an embedding.

        Returns:
            A list of floating-point values representing the text.
        """

        # Generate the embedding for the input text.
        # normalize_embeddings=True makes the vector normalized,
        # which is useful for similarity search.
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        # Convert the NumPy array returned by SentenceTransformer
        # into a standard Python list so it can be easily stored
        # and sent to Qdrant.
        return embedding.tolist()