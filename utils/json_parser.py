from typing import TypeVar, Optional
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)

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