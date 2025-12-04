from abc import ABC, abstractmethod
from typing import Dict, List
from model import BaseChunk, SearchResult

class VectorStore(ABC):
    """Base interface for vector database storage operations."""

    @abstractmethod
    def delete_collection(self):
        pass

    @abstractmethod
    def store_chunks(self, chunks: Dict[str, BaseChunk]):
        pass

    @abstractmethod
    def close(self):
        pass


class DenseSearchable(ABC):
    """Interface for databases that support dense vector search."""
    
    @abstractmethod
    def dense_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class SparseSearchable(ABC):
    """Interface for databases that support sparse vector search (e.g., BM25)."""
    
    @abstractmethod
    def sparse_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class HybridSearchable(ABC):
    """Interface for databases that support hybrid search (dense + sparse fusion)."""
    
    @abstractmethod
    def hybrid_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class ColbertSearchable(ABC):
    """Interface for databases that support late-interaction (ColBERT-style) search."""
    
    @abstractmethod
    def hybrid_search_with_colbert(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass


class CrossEncoderSearchable(ABC):
    """Interface for databases that support cross-encoder reranking."""
    
    @abstractmethod
    def hybrid_search_with_crossencoder(self, query: str, limit: int = 5) -> List[SearchResult]:
        pass