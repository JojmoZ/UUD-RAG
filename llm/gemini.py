from langchain_google_genai import ChatGoogleGenerativeAI
from .base import BaseLLM
from ragas.llms import LangchainLLMWrapper

class Gemini(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        
    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=1000
        )
    
    def get_ragas_llm(self):
        return LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=1000
        ))