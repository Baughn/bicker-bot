"""Local embeddings using sentence-transformers with CUDA."""

import logging
from functools import lru_cache
from typing import Any

import torch
from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get the best available device for embeddings."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU for embeddings")
    return device


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and cache the embedding model.

    Args:
        model_name: HuggingFace model name (e.g., 'nomic-ai/nomic-embed-text-v1.5')

    Returns:
        Loaded SentenceTransformer model on the best available device
    """
    device = get_device()
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    logger.info(f"Embedding model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    return model


class LocalEmbeddingFunction(EmbeddingFunction[list[str]]):
    """ChromaDB-compatible embedding function using local GPU."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model."""
        if self._model is None:
            self._model = get_embedding_model(self._model_name)
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            input: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not input:
            return []

        # nomic models expect a prefix for different use cases
        # "search_document: " for documents, "search_query: " for queries
        # We'll use search_document for storage and search_query for retrieval
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with query-optimized prefix.

        For nomic models, queries should have a different prefix than documents.
        """
        # nomic-embed-text uses prefixes
        if "nomic" in self._model_name.lower():
            prefixed = f"search_query: {query}"
        else:
            prefixed = query

        embedding = self.model.encode(
            [prefixed],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed documents with document-optimized prefix."""
        if not documents:
            return []

        # nomic-embed-text uses prefixes
        if "nomic" in self._model_name.lower():
            prefixed = [f"search_document: {doc}" for doc in documents]
        else:
            prefixed = documents

        embeddings = self.model.encode(
            prefixed,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
