from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pypdf.errors import PdfStreamError
import os
import re
import tempfile
import requests
from huggingface_hub import list_repo_files, hf_hub_download
from logger import Logger


class PDFLoader:
    def __init__(self, source: str, source_type: str = "local", hf_token: str = None):
        """
        Initialize PDFLoader
        
        Args:
            source: Either folder path (for local) or Hugging Face repo ID (for huggingface)
            source_type: "local" or "huggingface"
            hf_token: Hugging Face token for private repositories (required for private repos)
        """
        self.source = source
        self.source_type = source_type
        self.hf_token = hf_token
        self.pages = []
        
    async def load_langchain(self):
        if self.source_type == "local":
            await self._load_from_local()
        elif self.source_type == "huggingface":
            await self._load_from_huggingface()
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")
    
    async def _load_from_local(self):
        """Load PDFs from local folder"""
        Logger.log(f"[LOG] Loading PDFs from local folder: {self.source}")

        for file_name in os.listdir(self.source):
            file_path = os.path.join(self.source, file_name)
            if not file_name.lower().endswith(".pdf"):
                continue

            await self._load_single_pdf(file_path, file_name)
    
    async def _load_from_huggingface(self):
        """Load PDFs from Hugging Face repository"""
        Logger.log(f"[LOG] Loading PDFs from Hugging Face repo: {self.source}")
        
        repo_type_to_try = "dataset" 

        try:
            Logger.log(f"[LOG] Attempting to list files from repo as type: '{repo_type_to_try}'")
            files = list_repo_files(self.source, repo_type=repo_type_to_try, token=self.hf_token)
            
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            Logger.log(f"[LOG] Found {len(pdf_files)} PDF files in repository")
            
            for pdf_file in pdf_files:
                try:
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_path = temp_file.name
                        
                        downloaded_path = hf_hub_download(
                            repo_id=self.source,
                            filename=pdf_file,
                            token=self.hf_token,
                            repo_type=repo_type_to_try 
                        )
                        
                        import shutil
                        shutil.copy2(downloaded_path, temp_path)
                        
                        await self._load_single_pdf(temp_path, pdf_file)
                        
                        os.unlink(temp_path)
                        
                except Exception as e:
                    Logger.log(f"Failed to download or process {pdf_file}: {e}")
                    
        except Exception as e:
            Logger.log(f"CRITICAL: Failed to access Hugging Face repository: {e}")
            if "404" in str(e):
                 Logger.log("Got a 404 error. Please check your `repo_id` and `repo_type`.")
            if self.hf_token is None and "private" in str(e).lower():
                 Logger.log("Repository might be private. Please provide a Hugging Face token.")
            raise
    async def _load_single_pdf(self, file_path: str, file_name: str):
        """Load a single PDF file"""
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
        text = re.sub(r'PRESIDEN\s+REPUBLIK\s+INDONESIA', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SK\s+No\s+\d+\s*[A-Z]*', '', text)
        
        # Hapus nomor halaman
        text = re.sub(r'-\s*\d+\s*-', '', text)
        # Bersihkan karakter aneh
        text = re.sub(r"[^\w\s.,;:()'\-]", '', text)
        return text
        
    