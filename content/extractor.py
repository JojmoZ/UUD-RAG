from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pypdf.errors import PdfStreamError
import os
import re
from logger import Logger


class PDFLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.pages = []
        
    async def load_langchain(self):
        Logger.log(f"[LOG] Loading PDFs from folder: {self.folder_path}")

        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if not file_name.lower().endswith(".pdf"):
                continue

            try:
                loader = PyPDFLoader(file_path)
                async for page in loader.alazy_load():
                    page.page_content = self._clean_text(page.page_content)
                    self.pages.append(page)
            except PdfStreamError:
                Logger.log(f"Corrupt PDF detected, retrying with PyMuPDF: {file_name}")
                try:
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    if docs:
                        docs[0].page_content = self._clean_text(docs[0].page_content)
                        self.pages.append(docs[0])
                except Exception as e:
                    Logger.log(f"Failed to load {file_name} with PyMuPDF: {e}")
            except Exception as e:
                Logger.log(f"Unexpected error on {file_name}: {e}")
        
    def _clean_text(self,text: str) -> str:
        # Hapus header/footer umum
        text = re.sub(r'PRESIDEN\s+REPUBLIK\s+INDONESIA', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SK\s+No\s+\d+\s*[A-Z]*', '', text)
        
        # Hapus nomor halaman
        text = re.sub(r'-\s*\d+\s*-', '', text)
        # Bersihkan karakter aneh
        text = re.sub(r"[^\w\s.,;:()'\-]", '', text)
        return text
        
    