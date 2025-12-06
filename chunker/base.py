from langchain_core.documents import Document
from model import BaseChunk
from typing import Dict, List
import hashlib
import json
import os

class BaseChunker:
    def __init__(self, cache_dir: str = "./chunk_cache"):
        self.chunks : Dict[str, BaseChunk] = {}
        self.cache_dir = cache_dir
        self.document_chunks: Dict[str, List[str]] = {}  # Maps doc hash -> chunk IDs
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_document_hash(self, page: Document) -> str:
        """Generate unique hash for a document based on its content and metadata"""
        content = page.page_content
        source = page.metadata.get("source", "")
        page_num = page.metadata.get("page", 0)
        hash_input = f"{source}:{page_num}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, doc_hash: str) -> str:
        """Get cache file path for a document"""
        return os.path.join(self.cache_dir, f"{doc_hash}.json")
    
    def _load_cached_chunks(self, doc_hash: str) -> bool:
        """Load chunks from cache for a document. Returns True if loaded."""
        cache_path = self._get_cache_path(doc_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    chunk_ids = cached_data.get('chunk_ids', [])
                    chunks_data = cached_data.get('chunks', {})
                    chunk_type = cached_data.get('chunk_type', 'base')
                    
                    # Reconstruct chunk objects from JSON
                    chunks = self._reconstruct_chunks(chunks_data, chunk_type)
                    
                    # Add chunks to main collection
                    for chunk_id, chunk in chunks.items():
                        self.chunks[chunk_id] = chunk
                    
                    # Track document-to-chunks mapping
                    self.document_chunks[doc_hash] = chunk_ids
                    return True
            except Exception:
                pass
        return False
    
    def _save_chunks_to_cache(self, doc_hash: str, chunk_ids: List[str]):
        """Save chunks to cache for a document"""
        cache_path = self._get_cache_path(doc_hash)
        try:
            chunks_to_cache = {}
            for cid in chunk_ids:
                if cid in self.chunks:
                    # Convert Pydantic model to dict
                    chunks_to_cache[cid] = self.chunks[cid].model_dump()
            
            chunk_type = self._get_chunk_type()
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunk_ids': chunk_ids,
                    'chunks': chunks_to_cache,
                    'chunk_type': chunk_type
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def _get_chunk_type(self) -> str:
        """Get the chunk type for this chunker (to be overridden by subclasses)"""
        return 'base'
    
    def _reconstruct_chunks(self, chunks_data: Dict[str, dict], chunk_type: str) -> Dict[str, BaseChunk]:
        """Reconstruct chunk objects from JSON data (to be overridden by subclasses)"""
        return {}
    
    def get_uncached_documents(self, pages: List[Document]) -> List[Document]:
        """Filter out documents that are already cached"""
        uncached = []
        for page in pages:
            doc_hash = self._get_document_hash(page)
            if not self._load_cached_chunks(doc_hash):
                uncached.append(page)
        return uncached
    
    def get_chunks_for_database(self) -> List[BaseChunk]:
        """Get all chunks ready for database storage"""
        return list(self.chunks.values())
    
    def export_all_chunks_to_cache(self):
        """Export all chunks to cache after processing is complete"""
        from logger import Logger
        Logger.log(f"Exporting {len(self.document_chunks)} documents to cache...")
        
        for doc_hash, chunk_ids in self.document_chunks.items():
            self._save_chunks_to_cache(doc_hash, chunk_ids)
        
        Logger.log(f"✓ All chunks exported to {self.cache_dir}")

    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        raise NotImplementedError