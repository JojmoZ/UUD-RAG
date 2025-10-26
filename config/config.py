from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()
        self.GOOGLE_API_KEY = self.set_env("GOOGLE_API_KEY")
        self.GROQ_API_KEY = self.set_env("GROQ_API_KEY")
        
    
    def set_env(self, key) -> str:
        env = os.getenv(key)
        if env is None:
            raise ValueError(f"Environment variable {key} is not set.")
        return env
    