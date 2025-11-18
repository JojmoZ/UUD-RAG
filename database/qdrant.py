from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from logger import Logger
from qdrant_client.models import VectorParams, Distance, PointStruct
from embedder import BaseEmbedder

class Qdrant:
    def __init__(self,embedder : BaseEmbedder,google_api_key: str, qdrant_url : str = "http://localhost:6333", qdrant_api_key: str = None, collection_name: str = "documents"):
        Logger.log("qdrant url" + qdrant_url)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="gemini-embedding-001",
        #     google_api_key=google_api_key
        # )
        # model_name = "indobenchmark/indolegalbert-base"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name)
        self.embedder = embedder
        self._create_collection_if_not_exists()
        
    def _create_collection_if_not_exists(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                        size=self.embedder.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    )
            )
            Logger.log(f"Collection created: {self.collection_name}")
            return
        
    def add_documents(self, documents):
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=documents
            )
        except Exception as e:
            Logger.log(f"Error adding documents: {e}")
            raise
        
    def delete_collection(self):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            Logger.log(f"Error deleting collection: {e}")
            raise
    
    def get_info(self):
        try:
            info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            Logger.log(f"Error mendapatkan info collection: {e}")
            return {}
        
    def search(self, query: str, limit: int = 5):
        try:
            query_vector = self.embedder.encode(query)
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return search_result
        except Exception as e:
            Logger.log(f"Error searching documents: {e}")
            return []
        
    def store_chunks(self, chunks):
        points = self.embedder.generate_points_from_chunks(chunks)
        self.add_documents(points)
            
        
    def close(self):
        self.client.close()
    
        

        
        