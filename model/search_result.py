from pydantic import BaseModel
from typing import Any, Dict

class SearchResult(BaseModel):
    id: str
    payload: Dict[str, Any]
    score: float
