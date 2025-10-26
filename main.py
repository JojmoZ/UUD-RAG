from content.extractor import PDFLoader
from chunker import AgenticChunker
from config import Config
import asyncio
from logger import Logger

async def main():
    config = Config()
    loader = PDFLoader("./peraturan_pdfs")
    chunker = AgenticChunker(config.GOOGLE_API_KEY, config.GROQ_API_KEY)
    await loader.load_langchain()
    chunker.load_data_to_chunks(loader.pages)
    chunker.print_chunks()
    
if __name__ == "__main__":
    asyncio.run(main())


