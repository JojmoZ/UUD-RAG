from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class Point:
    id: str
    vector: np.ndarray
    payload: Dict[str, Any]
