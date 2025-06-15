from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract base for embedding generation"""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for text"""
        raise NotImplementedError("Subclasses must implement embed_text method")

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently"""
        raise NotImplementedError("Subclasses must implement embed_batch method")

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return embedding vector dimensions count"""
        raise NotImplementedError("Subclasses must implement get_dimensions method")
