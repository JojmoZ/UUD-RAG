from database import Qdrant
from logger import Logger
from langchain_core.prompts import ChatPromptTemplate
from llm import BaseLLM

class RAG:
    def __init__(self, database : Qdrant,llm: BaseLLM):
        self.database = database
        self.llm = llm
        
    def generate_answer(self, query: str, limit: int = 5):
        relevant_chunks =  self.database.search(query, limit)
        if not relevant_chunks:
            return {
                "answer": "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen hukum.",
                "sources": []
            }
        
        Logger.log(f"Found {len(relevant_chunks)} relevant chunks for query: '{query}'")
        Logger.log(f"Relevant chunks: {relevant_chunks}")
        
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(relevant_chunks):
            payload = chunk.payload
            context_parts.append(
                f"[Sumber {idx}] {payload['title']}\n"
                f"Ringkasan: {payload['summary']}\n"
                f"Konten:\n{payload['full_text']}\n"
            )
            sources.append({
                "index": idx,
                "title": payload['title'],
                "summary": payload['summary'],
                "score": chunk.score
            }) 
        
        context = "\n---\n".join(context_parts)
        
        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Anda adalah asisten hukum AI yang membantu menjawab pertanyaan tentang dokumen hukum Indonesia.
                
                Tugas Anda:
                1. Berikan jawaban yang akurat berdasarkan konteks dokumen hukum yang diberikan
                2. Jika informasi tidak ada dalam konteks, katakan dengan jelas
                3. Gunakan bahasa formal dan profesional
                4. Sebutkan pasal, undang-undang, atau referensi hukum yang spesifik jika ada
                5. Jika ada interpretasi, jelaskan dengan jelas bahwa itu adalah interpretasi
                
                Format jawaban:
                - Mulai dengan jawaban langsung
                - Berikan penjelasan detail jika diperlukan
                - Sebutkan dasar hukum atau sumber yang relevan
                - Jika ada ketidakpastian, sebutkan dengan jelas
                
                PENTING: Hanya gunakan informasi dari konteks yang diberikan. Jangan menambahkan informasi dari 
                pengetahuan umum Anda tanpa menyebutkan bahwa itu adalah informasi tambahan.
                """
            ),
            (
                "user",
                """
                Konteks dari dokumen hukum:
                
                {context}
                
                ---
                
                Pertanyaan: {question}
                
                Berikan jawaban yang lengkap dan akurat berdasarkan konteks di atas.
                """
            )
        ])
        
        try:
            
            answer = self.llm.answer(PROMPT, {"context": context, "question": query})
            payload = {
                "answer": answer,
                "sources": sources,
                "query": query
            }
            Logger.log(f"Jawaban berhasil di-generate untuk query: {payload}")
            
            return payload
        
        except Exception as e:
            Logger.log(f"Error generating answer: {e}")
            return {
                "answer": f"Maaf, terjadi error saat memproses pertanyaan: {str(e)}",
                "sources": sources
            }
        
        