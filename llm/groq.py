from .base import BaseLLM
from langchain_groq import ChatGroq


class Groq(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        
    def _initialize_llm(self):
        return ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key
        )