from langchain_core.prompts.chat import ChatPromptTemplate
from typing import Dict

class BaseLLM:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        if not api_key:
            raise ValueError("API key must be provided.")
        self.api_key = api_key
        self.model = self._initialize_llm()

    def answer(self, PROMPT: ChatPromptTemplate, input : Dict) -> str:
        runnable = PROMPT | self.model
        return runnable.invoke(input).content
        
    def _initialize_llm(self):
        raise NotImplementedError("Subclasses must implement this method.")