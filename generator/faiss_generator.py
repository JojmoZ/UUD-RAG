from database.faiss_db import FAISS
from logger import Logger
from llm import BaseLLM
from .base import BaseGenerator
from typing import List

class FAISSGenerator(BaseGenerator):
    def __init__(self, database: FAISS, llm: BaseLLM):
        """
        Initialize FAISS-based RAG generator
        
        Args:
            database: FAISS database instance
            llm: Language model for answer generation
        """
        super().__init__(database, llm)
        
    def generate_answer(self, query: str, limit: int = 5):
        """
        Generate answer using FAISS database for retrieval
        
        Args:
            query: User query/question
            limit: Number of chunks to retrieve
            
        Returns:
            Dict containing answer and sources
        """
        try:
            relevant_chunks = self.database.search(query, limit)
            
            if not relevant_chunks:
                return {
                    "answer": "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen hukum.",
                    "sources": []
                }
            
            Logger.log(f"Found {len(relevant_chunks)} relevant chunks for query: '{query}'")
            Logger.log(f"Relevant chunks scores: {[chunk.score for chunk in relevant_chunks]}")
            
            context_parts = []
            sources = []
            
            for idx, chunk in enumerate(relevant_chunks):
                payload = chunk.payload
                
                context_parts.append(payload['full_text'])
                
                source_info = {
                    "score": chunk.score,
                    "chunk_id": payload.get('chunk_id', chunk.id)
                }
                
                if 'source' in payload:
                    source_info.update({
                        "source": payload['source'],
                        "page": payload['page'],
                        "page_label": payload['page_label'],
                        "total_pages": payload['total_pages']
                    })
                
                if 'title' in payload:
                    source_info.update({
                        "title": payload['title'],
                        "summary": payload['summary'],
                        "index": payload['index']
                    })
                
                sources.append(source_info)
            
            context = "\n---\n".join(context_parts)
            prompt = self.generate_prompt(context, query)
            answer = self.llm.answer(prompt, {"context": context, "question": query})
            
            result = {
                "answer": answer,
                "sources": sources,
                "query": query,
                "retrieval_method": "faiss_dense"
            }
            
            Logger.log(f"Successfully generated answer for query using FAISS")
            return result
        
        except Exception as e:
            Logger.log(f"Error generating answer with FAISS: {e}")
            return {
                "answer": f"Maaf, terjadi error saat memproses pertanyaan: {str(e)}",
                "sources": sources if 'sources' in locals() else [],
                "query": query,
                "retrieval_method": "faiss_dense"
            }