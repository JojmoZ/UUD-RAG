from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct


class BaseEmbedder:
    def __init__(self):
        self.embedder = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")

    def generate_points_from_chunks(self, chunks) -> list[PointStruct]:
        return NotImplementedError
    
    def encode(self, text):
        return self.embedder.encode(text)

    def get_sentence_embedding_dimension(self):
        return self.embedder.get_sentence_embedding_dimension()
    
    