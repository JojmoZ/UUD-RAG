from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from logger import Logger
import uuid
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional
from langchain_core.documents import Document
from pydantic import BaseModel, ValidationError
from typing import TypeVar, Type, Optional
import json
import os
import hashlib
from llm.base import BaseLLM

T = TypeVar("T", bound=BaseModel)

class AgenticChunker:
    def __init__(self, llm: BaseLLM, cache_dir: str = "./chunk_cache"):
        
        self.llm = llm
        
        self.chunks = {}
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_document_hash(self, text: str) -> str:
        """Generate a hash for a document to use as cache key"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_cache_path(self, doc_hash: str) -> str:
        """Get the cache file path for a document hash"""
        return os.path.join(self.cache_dir, f"{doc_hash}.json")
    
    
    def save_chunks(self, filepath: str = None):
        """Save all chunks to a JSON file"""
        if filepath is None:
            filepath = os.path.join(self.cache_dir, "all_chunks.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        Logger.log(f"Saved {len(self.chunks)} chunks to {filepath}")
    
    def load_chunks(self, filepath: str = None) -> bool:
        """Load chunks from a JSON file"""
        if filepath is None:
            filepath = os.path.join(self.cache_dir, "all_chunks.json")
        
        if not os.path.exists(filepath):
            Logger.log(f"No cache file found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            Logger.log(f"Loaded {len(self.chunks)} chunks from {filepath}")
            return True
        except Exception as e:
            Logger.log(f"Error loading chunks: {e}")
            return False
    
    def load_data_to_chunks(self, pages: list[Document], use_cache: bool = True):
        """
        Load documents and chunk them. Uses cache if available.
        Saves after EVERY page is processed.
        
        Args:
            pages: List of Document objects to chunk
            use_cache: If True, will check cache before processing
        """
        for idx, page in enumerate(pages):
            doc_hash = self._get_document_hash(page.page_content)
            cache_path = self._get_cache_path(doc_hash)
            
            if use_cache and os.path.exists(cache_path):
                Logger.log(f"[Page {idx+1}/{len(pages)}] Loading cached chunks for document hash: {doc_hash[:8]}...")
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_chunks = json.load(f)
                    
                    for chunk_id, chunk_data in cached_chunks.items():
                        if chunk_id not in self.chunks:
                            self.chunks[chunk_id] = chunk_data
                    
                    Logger.log(f"Loaded {len(cached_chunks)} cached chunks")
                    
                    self.save_chunks()
                    continue
                except Exception as e:
                    Logger.log(f"Error loading cache, will reprocess: {e}")
            
            Logger.log(f"[Page {idx+1}/{len(pages)}] Processing new document (hash: {doc_hash[:8]}...)")
            chunks_before = len(self.chunks)
            self._generate_propositions(page.page_content)
            chunks_after = len(self.chunks)
            
            new_chunks = {k: v for k, v in list(self.chunks.items())[chunks_before:chunks_after]}
            
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(new_chunks, f, indent=2, ensure_ascii=False)
                Logger.log(f"Cached {len(new_chunks)} chunks for this document")
                
                # SAVE ALL CHUNKS after every page
                self.save_chunks()
                Logger.log(f"Saved all {len(self.chunks)} chunks to disk")
                
            except Exception as e:
                Logger.log(f"Error saving cache: {e}")
        
        Logger.log(f"Generated propositions for {len(pages)} pages. Total chunks: {len(self.chunks)}")
    
    def clear_cache(self):
        """Clear all cached chunk files"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            Logger.log("Cache cleared")
    
    def parse_json_response(self, result: str, model: type[T]) -> Optional[T]:
        """Clean up a JSON-like string returned from an LLM and parse it into a Pydantic model."""
        cleaned = result.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        try:
            return model.model_validate_json(cleaned)
        except ValidationError as e:
            print("[ERROR] JSON validation failed:", e)
            return None
        except Exception as e:
            print("[ERROR] Failed to parse JSON:", e)
            return None
        
    def _generate_propositions(self, text: str) -> list[str]:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Uraikan "Konten" dokumen hukum menjadi proposisi yang jelas dan sederhana, pastikan proposisi tersebut 
                    dapat dipahami di luar konteks aslinya.
                    
                    1. Pisahkan kalimat majemuk menjadi kalimat-kalimat sederhana. Pertahankan frasa asli dari input 
                    sebisa mungkin.
                    
                    2. Untuk setiap entitas bernama (nama orang, lembaga, peraturan, pasal, dll.) yang disertai informasi 
                    deskriptif tambahan, pisahkan informasi ini menjadi proposisi tersendiri.
                    
                    3. Decontekstualisasikan proposisi dengan menambahkan modifier yang diperlukan pada kata benda atau 
                    seluruh kalimat, dan ganti kata ganti (misalnya, "ini", "itu", "ia", "mereka", "tersebut") dengan 
                    nama lengkap entitas yang dirujuk.
                    
                    4. Kembalikan output dalam format ini:
                    {{
                        "propositions": [
                        "proposisi 1",
                        "proposisi 2",
                        ...
                        ]
                    }}

                    JANGAN sertakan penjelasan apapun, markdown code fences, atau teks di luar JSON.

                    Contoh:

                    Input: Judul: Undang-Undang Cipta Kerja. Bagian: Ketentuan Umum, Definisi Pekerja. Konten:
                    Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja disahkan pada tanggal 2 November 2020 
                    oleh Presiden Joko Widodo. Undang-undang ini mengatur berbagai aspek ketenagakerjaan termasuk 
                    definisi pekerja. Dalam undang-undang ini, pekerja didefinisikan sebagai setiap orang yang bekerja 
                    dengan menerima upah atau imbalan dalam bentuk lain. Pekerja dapat bekerja berdasarkan perjanjian 
                    kerja waktu tertentu atau perjanjian kerja waktu tidak tertentu. Pengusaha wajib memberikan jaminan 
                    sosial kepada pekerja sesuai ketentuan peraturan perundang-undangan.
                    
                    Output: [
                    "Undang-Undang Nomor 11 Tahun 2020 adalah tentang Cipta Kerja.",
                    "Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja disahkan pada tanggal 2 November 2020.",
                    "Undang-Undang Nomor 11 Tahun 2020 tentang Cipta Kerja disahkan oleh Presiden Joko Widodo.",
                    "Undang-Undang Cipta Kerja mengatur berbagai aspek ketenagakerjaan.",
                    "Undang-Undang Cipta Kerja mengatur definisi pekerja.",
                    "Dalam Undang-Undang Cipta Kerja, pekerja didefinisikan sebagai setiap orang yang bekerja dengan menerima upah atau imbalan dalam bentuk lain.",
                    "Pekerja dapat bekerja berdasarkan perjanjian kerja waktu tertentu.",
                    "Pekerja dapat bekerja berdasarkan perjanjian kerja waktu tidak tertentu.",
                    "Pengusaha wajib memberikan jaminan sosial kepada pekerja.",
                    "Kewajiban pengusaha memberikan jaminan sosial sesuai ketentuan peraturan perundang-undangan."
                    ]
                    """
                ),
                ("user", "Uraikan konten hukum berikut:: {input}"),
            ]
        )
        result = self.llm.answer(PROMPT, {"input": text})
        
        class Sentences (BaseModel):
            propositions: list[str]
            
        sentences = self.parse_json_response(result, Sentences)
        
        if sentences is None:
            Logger.log("No propositions generated.")
            return []
        
        for proposition in sentences.propositions:
            self.add_proposition(proposition)
            
        return sentences.propositions
    
    def _update_chunk_summary(self, chunk):
        """If you add a new proposition to a chunk, you may want to update the summary or else they could get stale"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Sebuah proposisi baru saja ditambahkan ke salah satu chunk Anda. Anda harus membuat ringkasan sangat 
                    singkat dalam 1 kalimat yang menginformasikan pembaca tentang isi kelompok chunk tersebut.

                    Ringkasan yang baik akan menjelaskan tentang apa chunk tersebut, dan memberikan instruksi klarifikasi 
                    tentang apa yang harus ditambahkan ke chunk tersebut.

                    Anda akan diberikan sekelompok proposisi yang ada dalam chunk dan ringkasan chunk saat ini.

                    Ringkasan Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang Pasal 1, 
                    generalisasikan ke "ketentuan pasal". Atau jika tentang hak pekerja, generalisasikan ke "hak dan 
                    kewajiban tenaga kerja".

                    Contoh:
                    Input: Proposisi: Pasal 27 UUD 1945 menjamin hak setiap warga negara untuk mendapatkan pekerjaan
                    Output: Chunk ini berisi informasi tentang ketentuan hak warga negara terkait pekerjaan dalam UUD 1945.

                    Hanya berikan ringkasan chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Proposisi dalam Chunk:\n{proposition}\n\nRingkasan chunk saat ini:\n{current_summary}"),
            ]
        )

        new_chunk_summary = self.llm.answer(PROMPT, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary']
        })

        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        """If you add a new proposition to a chunk, you may want to update the title or else it can get stale"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Sebuah proposisi baru saja ditambahkan ke salah satu chunk Anda. Anda harus membuat judul chunk yang 
                    sangat singkat dan diperbarui yang menginformasikan pembaca tentang isi kelompok chunk tersebut.

                    Judul yang baik akan menjelaskan tentang apa chunk tersebut.

                    Anda akan diberikan sekelompok proposisi yang ada dalam chunk, ringkasan chunk, dan judul chunk.

                    Judul Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang Pasal 1, 
                    generalisasikan ke "Ketentuan Pasal". Atau jika tentang putusan pengadilan, generalisasikan ke 
                    "Yurisprudensi & Putusan".

                    Contoh:
                    Input: Ringkasan: Chunk ini berisi informasi tentang tanggal dan waktu pemberlakuan peraturan
                    Output: Tanggal Berlaku Peraturan

                    Hanya berikan judul chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Proposisi dalam Chunk:\n{proposition}\n\nRingkasan chunk:\n{current_summary}\n\nJudul chunk saat ini:\n{current_title}"),
            ]
        )
        
        new_chunk_title = self.llm.answer(PROMPT, {
            "proposition": "\n".join(chunk['propositions']),
            "current_summary" : chunk['summary'],
            "current_title" : chunk['title']
        })

        return new_chunk_title
    
    def _get_new_chunk_summary(self, proposition: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Anda harus membuat ringkasan sangat singkat dalam 1 kalimat yang menginformasikan pembaca tentang 
                    isi kelompok chunk tersebut.

                    Ringkasan yang baik akan menjelaskan tentang apa chunk tersebut, dan memberikan instruksi klarifikasi 
                    tentang apa yang harus ditambahkan ke chunk tersebut.

                    Anda akan diberikan proposisi yang akan masuk ke chunk baru. Chunk baru ini memerlukan ringkasan.

                    Ringkasan Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang sanksi pidana, 
                    generalisasikan ke "ketentuan sanksi". Atau jika tentang januari 2020, generalisasikan ke "tanggal 
                    dan waktu".

                    Contoh:
                    Input: Proposisi: Pasal 88 mengatur tentang sanksi pidana penjara minimal 5 tahun
                    Output: Chunk ini berisi informasi tentang ketentuan sanksi pidana dalam peraturan.

                    Hanya berikan ringkasan chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Tentukan ringkasan chunk baru dimana proposisi ini akan masuk:\n{proposition}"),
            ]
        )

        
        new_chunk_summary = self.llm.answer(PROMPT, {
            "proposition": proposition
        })

        return new_chunk_summary
        
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Anda adalah pengelola kumpulan chunk yang merepresentasikan kelompok proposisi hukum yang membahas 
                    topik serupa.
                    
                    Anda harus membuat judul chunk yang sangat singkat (beberapa kata) yang menginformasikan pembaca 
                    tentang isi kelompok chunk tersebut.

                    Judul chunk yang baik adalah singkat tetapi mencakup tentang apa chunk tersebut.

                    Anda akan diberikan ringkasan dari chunk yang memerlukan judul.

                    Judul Anda harus mengantisipasi generalisasi. Jika Anda mendapat proposisi tentang UU Ketenagakerjaan, 
                    generalisasikan ke "Peraturan Ketenagakerjaan". Atau jika tentang Maret 2020, generalisasikan ke 
                    "Tanggal & Waktu".

                    Contoh:
                    Input: Ringkasan: Chunk ini berisi informasi tentang tanggal dan waktu pemberlakuan undang-undang
                    Output: Tanggal & Waktu Pemberlakuan

                    Hanya berikan judul chunk baru, tidak ada yang lain.
                    """,
                ),
                ("user", "Tentukan judul chunk dimana ringkasan ini termasuk:\n{summary}"),
            ]
        )

        new_chunk_title = self.llm.answer(PROMPT, {
            "summary": summary
        })
        return new_chunk_title
    
    def _create_chunk(self, proposition: str) -> str:
        id = str(uuid.uuid4())
        summary = self._get_new_chunk_summary(proposition)
        title = self._get_new_chunk_title(summary)
        
        self.chunks[id] = {
            "id": id,
            "title": title,
            "summary": summary,
            "propositions": [proposition],
            "index" : len(self.chunks)
        }
        
        Logger.log(f"Created new chunk with ID: {id}, Title: {title}, Summary: {summary}")
        
    def get_chunks(self):
        chunks = "\n".join([f"Chunk ID: {chunk['id']}, Judul: {chunk['title']}, Ringkasan: {chunk['summary']}" for chunk in self.chunks.values()])
        return chunks
    
    def _find_similar_chunk(self, proposition: str):
        chunks = self.get_chunks()
        
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Tentukan apakah "Proposisi" hukum harus masuk ke salah satu chunk yang sudah ada atau tidak.

                    Sebuah proposisi harus masuk ke chunk jika makna, arah, atau maksudnya serupa.
                    Tujuannya adalah mengelompokkan proposisi dan chunk yang serupa.

                    Jika Anda berpikir proposisi harus digabungkan dengan chunk, kembalikan:
                    {{
                        "chunk_id": "<chunk_id>"
                    }}
                    
                    Jika Anda tidak berpikir proposisi harus digabungkan dengan chunk yang sudah ada, kembalikan:
                    {{
                        "chunk_id": null
                    }}
                    
                    JANGAN sertakan penjelasan, markdown, tanda kutip, atau teks tambahan. Hanya kembalikan objek JSON.
                    
                    Contoh:
                    Input:
                        - Proposisi: "Pasal 27 UUD 1945 mengatur tentang hak warga negara untuk mendapatkan pekerjaan"
                        - Chunk Saat Ini:
                            - Chunk ID: 2n4l3d
                            - Nama Chunk: Ketentuan Pidana
                            - Ringkasan Chunk: Berisi informasi tentang sanksi pidana dan denda

                            - Chunk ID: 93833k
                            - Nama Chunk: Hak Warga Negara
                            - Ringkasan Chunk: Daftar hak-hak warga negara yang diatur dalam UUD 1945
                            
                    Output: {{
                        "chunk_id": "93833k"
                    }}
                    """,
                ),
                ("user", "Chunk Saat Ini:\n--Awal chunk saat ini--\n{current_chunk_outline}\n--Akhir chunk saat ini--"),
                ("user", "Tentukan apakah proposisi berikut harus masuk ke salah satu chunk yang diuraikan:\n{proposition}"),
            ]
        )
        
        chunk_found = self.llm.answer(PROMPT, {
            "proposition": proposition,
            "current_chunk_outline": chunks
        })
        
        Logger.log(f"Chunk found: {chunk_found}")
        
        class ChunkMatch(BaseModel):
            chunk_id: Optional[str] = None
            
        chunk_found = self.parse_json_response(chunk_found, ChunkMatch)
        
        if chunk_found is None or chunk_found.chunk_id not in self.chunks:
            return None
        return chunk_found.chunk_id
    
    def add_proposition_to_chunk(self, chunk_id: str, proposition: str):
        self.chunks[chunk_id]['propositions'].append(proposition)
        
        self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])
        self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
        
    def add_proposition(self, proposition: str):
        Logger.log(f"Adding proposition to chunker: {proposition}")
        
        if len(self.chunks) == 0:
            self._create_chunk(proposition)
            return
        
        # Find the most similar chunk
        most_similar_chunk_id = self._find_similar_chunk(proposition)
        if most_similar_chunk_id is None:
            self._create_chunk(proposition)
            return
        
        self.add_proposition_to_chunk(most_similar_chunk_id, proposition)
        
    def print_chunks(self):
        for chunk in self.chunks.values():
            print(f"Chunk ID: {chunk['id']}")
            print(f"Judul Chunk: {chunk['title']}")
            print(f"Ringkasan Chunk: {chunk['summary']}")
            print(f"Proposisi Chunk: {chunk['propositions']}")
            print("")