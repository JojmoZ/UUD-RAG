import re
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pypdf.errors import PdfStreamError
from logger import Logger


class BaseLoader:
    def __init__(self, source: str):
        self.source = source
        self.pages = []

    def _clean_text(self,text: str) -> str:
        text = re.sub(r'PRESIDEN\s+REPUBLIK\s+INDONESIA', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SK\s+No\s+\d+\s*[A-Z]*', '', text)
        
        # Hapus nomor halaman
        text = re.sub(r'-\s*\d+\s*-', '', text)
        # Bersihkan karakter aneh
        text = re.sub(r"[^\w\s.,;:()'\-]", '', text)
        return text
    
    def load_data(self):
        raise NotImplementedError

    async def _load_single_pdf(self, file_path: str, file_name: str):
        """Load a single PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            pages_loaded = []
            async for page in loader.alazy_load():
                page.page_content = self._clean_text(page.page_content)
                pages_loaded.append(page)
            self.pages.extend(pages_loaded)
            
        except PdfStreamError:
            Logger.log(f"Corrupt PDF detected, retrying with PyMuPDF: {file_name}")
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.page_content = self._clean_text(doc.page_content)
                    self.pages.append(doc)
            except Exception as e:
                Logger.log(f"Failed to load {file_name} with PyMuPDF: {e}")
        except Exception as e:
            Logger.log(f"Unexpected error on {file_name}: {e}")