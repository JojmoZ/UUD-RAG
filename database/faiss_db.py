import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from logger import Logger
from model import BaseChunk
from typing import Dict, List, Tuple, Optional

class ScoredPoint:
    """Custom scored point class to match Qdrant's interface"""
    def __init__(self, id: str, payload: dict, score: float):
        self.id = id
        self.payload = payload
        self.score = score

class FAISS:
    def __init__(self, 
                 index_path: str = "./faiss_index", 
                 dense_model_name: str = "LazarusNLP/all-indo-e5-small-v4",
                 collection_name: str = "documents"):
        """
        Initialize FAISS database
        
        Args:
            index_path: Path to store FAISS index files
            dense_model_name: Name of the sentence transformer model
            collection_name: Collection identifier for this database
        """
        self.index_path = index_path
        self.collection_name = collection_name
        self.dense_model_name = dense_model_name
        
        Logger.log(f"Loading dense model: {dense_model_name}")
        self.dense_model = SentenceTransformer(dense_model_name)
        self.embedding_dim = self.dense_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.id_to_payload = {}
        self.id_to_index = {}
        self.index_to_id = {}
        
        os.makedirs(index_path, exist_ok=True)
        
        self._load_index()
        
        Logger.log(f"FAISS database initialized with collection: {collection_name}")
        
    def _create_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        Logger.log(f"Created new FAISS index with dimension: {self.embedding_dim}")
        
    def _load_index(self):
        """Load existing FAISS index from disk"""
        index_file = os.path.join(self.index_path, f"{self.collection_name}.index")
        metadata_file = os.path.join(self.index_path, f"{self.collection_name}_metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.id_to_payload = metadata['id_to_payload']
                    self.id_to_index = metadata['id_to_index']
                    self.index_to_id = metadata['index_to_id']
                
                Logger.log(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                Logger.log(f"Error loading existing index: {e}")
                self._create_index()
        else:
            self._create_index()
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            index_file = os.path.join(self.index_path, f"{self.collection_name}.index")
            metadata_file = os.path.join(self.index_path, f"{self.collection_name}_metadata.pkl")
            
            faiss.write_index(self.index, index_file)
            
            metadata = {
                'id_to_payload': self.id_to_payload,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
            Logger.log(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            Logger.log(f"Error saving index: {e}")
            raise
    
    def add_documents(self, points):
        """Add documents to FAISS index (maintaining compatibility with Qdrant interface)"""
        try:
            vectors = []
            for point in points:
                if hasattr(point, 'vector') and isinstance(point.vector, dict):
                    dense_vector = point.vector.get('dense')
                elif hasattr(point, 'vector'):
                    dense_vector = point.vector
                else:
                    raise ValueError(f"Invalid point structure: {point}")
                
                if dense_vector is None:
                    raise ValueError("Dense vector not found in point")
                
                vectors.append(dense_vector)
                
                point_id = str(point.id)
                current_index = len(self.id_to_index)
                
                self.id_to_payload[point_id] = point.payload
                self.id_to_index[point_id] = current_index
                self.index_to_id[current_index] = point_id
            
            vectors_np = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_np)  
            self.index.add(vectors_np)
            self._save_index()
            
            Logger.log(f"Added {len(points)} documents to FAISS index")
            
        except Exception as e:
            Logger.log(f"Error adding documents to FAISS: {e}")
            raise
    
    def dense_search(self, query: str, limit: int = 5) -> List[ScoredPoint]:
        """Search using dense embeddings only"""
        try:
            if self.index.ntotal == 0:
                Logger.log("No documents in FAISS index")
                return []
            
            query_vector = self.dense_model.encode([query])
            query_vector = np.array(query_vector, dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, min(limit, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    point_id = self.index_to_id[idx]
                    payload = self.id_to_payload[point_id]
                    results.append(ScoredPoint(id=point_id, payload=payload, score=float(score)))
            
            Logger.log(f"FAISS dense search found {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            Logger.log(f"Error in FAISS dense search: {e}")
            return []
    
    def search(self, query: str, limit: int = 5) -> List[ScoredPoint]:
        """Default search method (uses dense search for FAISS)"""
        return self.dense_search(query, limit)
    
    def store_chunks(self, chunks: Dict[str, BaseChunk]):
        """Store chunks in FAISS database"""
        try:
            texts = [chunk.get_context() for chunk in chunks.values()]
            embeddings = self.dense_model.encode(texts)
            
            points = []
            for i, (chunk_id, chunk) in enumerate(chunks.items()):
                class Point:
                    def __init__(self, id, vector, payload):
                        self.id = id
                        self.vector = vector
                        self.payload = payload
                
                point = Point(
                    id=chunk_id,
                    vector=embeddings[i],
                    payload=chunk.get_payload()
                )
                points.append(point)
            
            self.add_documents(points)
            Logger.log(f"Stored {len(chunks)} chunks in FAISS database")
            
        except Exception as e:
            Logger.log(f"Error storing chunks in FAISS: {e}")
            raise
    
    def delete_collection(self):
        """Delete the FAISS index and metadata"""
        try:
            index_file = os.path.join(self.index_path, f"{self.collection_name}.index")
            metadata_file = os.path.join(self.index_path, f"{self.collection_name}_metadata.pkl")
            
            if os.path.exists(index_file):
                os.remove(index_file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            self._create_index()
            self.id_to_payload = {}
            self.id_to_index = {}
            self.index_to_id = {}
            
            Logger.log(f"Deleted FAISS collection: {self.collection_name}")
            
        except Exception as e:
            Logger.log(f"Error deleting FAISS collection: {e}")
            raise
    
    def get_info(self):
        """Get information about the FAISS database"""
        try:
            return {
                "name": self.collection_name,
                "vectors_count": self.index.ntotal if self.index else 0,
                "points_count": self.index.ntotal if self.index else 0,
                "status": "ready" if self.index else "not_initialized",
                "embedding_dimension": self.embedding_dim,
                "model_name": self.dense_model_name
            }
        except Exception as e:
            Logger.log(f"Error getting FAISS info: {e}")
            return {}
    
    def close(self):
        """Clean up resources"""
        try:
            self._save_index()
            Logger.log("FAISS database closed successfully")
        except Exception as e:
            Logger.log(f"Error closing FAISS database: {e}")