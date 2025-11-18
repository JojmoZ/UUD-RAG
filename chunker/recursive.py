from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from .base import BaseChunker

class RecursiveChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 50):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        self.chunks = self.text_splitter.split_documents(pages)
        print(f"Generated {len(self.chunks)} chunks")
        print(self.chunks)