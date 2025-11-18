from langchain_core.documents import Document

class BaseChunker:
    def __init__(self):
        self.chunks = {}

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        raise NotImplementedError