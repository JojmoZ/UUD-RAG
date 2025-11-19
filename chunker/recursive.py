from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from .base import BaseChunker
import uuid

class RecursiveChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 50):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        chunks = self.text_splitter.split_documents(pages)
        for chunk in chunks:
            id = str(uuid.uuid4())
            metadata = chunk.metadata if chunk.metadata else {}

            payload = {
                "chunk_id": id,
                "full_text": chunk.page_content,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "total_pages": metadata.get("total_pages"),
                "page_label": metadata.get("page_label")
            }
        
            self.chunks[id] = payload