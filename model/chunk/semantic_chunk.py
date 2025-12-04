from .base_chunk import BaseChunk

class SemanticChunk(BaseChunk):
    content: str
    source: str
    page: int
    total_pages: int
    page_label: str
    semantic_score: float
    boundary_type: str
    
    def get_context(self) -> str:
        return self.content
    
    def get_payload(self) -> dict:
        return {
            "chunk_id": self.id,
            "full_text": self.content,
            "source": self.source,
            "page": self.page,
            "total_pages": self.total_pages,
            "page_label": self.page_label,
            "semantic_score": self.semantic_score,
            "boundary_type": self.boundary_type,
            "chunk_type": "semantic"
        }