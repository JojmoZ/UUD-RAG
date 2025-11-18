from .base import BaseEmbedder
from qdrant_client.models import PointStruct
from logger import Logger


class AgenticEmbedder(BaseEmbedder):
    def __init__(self):
        super().__init__()

    def generate_points_from_chunks(self, chunks) -> list[PointStruct]:
        points = []
        for chunk_id, data in chunks.items():
            content = "\n".join(data['propositions'])
            full_text =  f"Judul: {data['title']}\nRingkasan: {data['summary']}\n\nKonten:\n{content}"
            
            try:
                vector = self.encode(full_text)
                payload={
                        "chunk_id": chunk_id,
                        "title": data['title'],
                        "summary": data['summary'],
                        "propositions": data['propositions'],
                        "full_text": content,
                        "index": data['index']
                    }
                point = PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            except Exception as e:
                Logger.log(f"Error embedding chunk {chunk_id}: {e}")
                
        return points