from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from sentence_transformers import SentenceTransformer
from .base import BaseChunker
import uuid
from model.chunk.semantic_chunk import SemanticChunk
from typing import Dict, List
from logger import Logger

class SemanticChunker(BaseChunker):
    def __init__(self, 
                 embedding_model_name: str = "LazarusNLP/all-indo-e5-small-v4",
                 breakpoint_threshold_type: str = "percentile",
                 breakpoint_threshold_amount: float = 95.0,
                 number_of_chunks: int = None,
                 cache_dir: str = "./chunk_cache"):
        super().__init__(cache_dir=cache_dir)
        
        Logger.log(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        
        class SentenceTransformerEmbeddings:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        
        embeddings_wrapper = SentenceTransformerEmbeddings(self.embedding_model)
        
        self.text_splitter = LangChainSemanticChunker(
            embeddings=embeddings_wrapper,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks
        )
        
        Logger.log(f"Semantic chunker initialized with {breakpoint_threshold_type} threshold: {breakpoint_threshold_amount}")

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        try:
            if use_cache:
                uncached_pages = self.get_uncached_documents(pages)
                if len(uncached_pages) < len(pages):
                    Logger.log(f"Loaded {len(pages) - len(uncached_pages)} documents from cache")
                pages = uncached_pages
            
            if not pages:
                Logger.log("All documents already cached")
                return
            
            total_pages = len(pages)
            Logger.log(f"Processing {total_pages} uncached documents with semantic chunking...")
            
            # Process each document separately to maintain per-document caching
            for idx, page in enumerate(pages, 1):
                doc_hash = self._get_document_hash(page)
                chunk_ids = []
                
                # Split this specific page
                split_docs = self.text_splitter.split_documents([page])
                
                for doc in split_docs:
                    id = str(uuid.uuid4())
                    metadata = doc.metadata or {}

                    chunk_obj = SemanticChunk(
                        id=id,
                        content=doc.page_content,
                        source=metadata.get("source"),
                        page=metadata.get("page"),
                        total_pages=metadata.get("total_pages"),
                        page_label=metadata.get("page_label"),
                        semantic_score=0.0,  # Default score
                        boundary_type="semantic"  # Semantic boundary type
                    )

                    self.chunks[id] = chunk_obj
                    chunk_ids.append(id)
                
                # Track document-to-chunks mapping (will be saved later)
                self.document_chunks[doc_hash] = chunk_ids
                
                # Show progress every 100 pages
                if idx % 100 == 0 or idx == total_pages:
                    Logger.log(f"Progress: {idx}/{total_pages} pages processed ({idx*100//total_pages}%)")
            
            Logger.log(f"Semantic chunker created {len(self.chunks)} total chunks")
            
            # Export all chunks to cache after processing all documents
            self.export_all_chunks_to_cache()
            
        except Exception as e:
            Logger.log(f"Error in semantic chunking: {e}")
            raise
    
    def get_chunker_info(self):
        return {
            "chunker_type": "semantic",
            "embedding_model": self.embedding_model_name,
            "breakpoint_threshold_type": self.text_splitter.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.text_splitter.breakpoint_threshold_amount,
            "total_chunks": len(self.chunks)
        }
    
    def _get_chunk_type(self) -> str:
        return 'semantic'
    
    def _reconstruct_chunks(self, chunks_data: Dict[str, dict], chunk_type: str) -> Dict[str, SemanticChunk]:
        """Reconstruct SemanticChunk objects from JSON data"""
        chunks = {}
        for chunk_id, chunk_dict in chunks_data.items():
            chunks[chunk_id] = SemanticChunk(**chunk_dict)
        return chunks