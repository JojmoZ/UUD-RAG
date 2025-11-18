from .base import BaseEmbedder
from qdrant_client.models import PointStruct
from logger import Logger


class RecursiveEmbedder(BaseEmbedder):
    def __init__(self):
        super().__init__()

    def generate_points_from_chunks(self, chunks) -> list[PointStruct]:
        points = []
        for chunk_id, data in chunks.items():

            content = data.page_content

            try:
                vector = self.encode(content)

                metadata = data.metadata if data.metadata else {}

                payload = {
                    "chunk_id": chunk_id,
                    "full_text": content,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "total_pages": metadata.get("total_pages"),
                    "page_label": metadata.get("page_label")
                }

                point = PointStruct(
                    id=str(chunk_id),
                    vector=vector,
                    payload=payload
                )
                points.append(point)

            except Exception as e:
                Logger.log(f"Error embedding chunk {chunk_id}: {e}")

        return points
