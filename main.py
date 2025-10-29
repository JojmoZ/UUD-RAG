from content.extractor import PDFLoader
from chunker import AgenticChunker
from config import Config
import asyncio
from logger import Logger
from llm import Gemini, Groq
from database import Qdrant
from generator import RAG

async def main():
    config = Config()
    loader = PDFLoader("RM-TextMining/UU-RAG", source_type="huggingface")
    gemini = Gemini("gemini-2.5-pro", config.GOOGLE_API_KEY)
    groq = Groq("meta-llama/llama-guard-4-12b",config.GROQ_API_KEY)
    chunker = AgenticChunker(groq)
    await loader.load_langchain()
    
    chunker.load_chunks()
    # chunker.print_chunks()
    db = Qdrant(config.GOOGLE_API_KEY,config.QDRANT_HOST, config.QDRANT_API_KEY, "peraturan_hukum")
    # db.store_chunks(chunker.chunks)
    Logger.log("All chunks stored in Qdrant database.")
    rag = RAG(db, gemini)
    answer = rag.generate_answer("Apa saja ketentuan mengenai sanksi administratif dalam peraturan tersebut?")
    Logger.log(answer)
    
    
if __name__ == "__main__":
    asyncio.run(main())


