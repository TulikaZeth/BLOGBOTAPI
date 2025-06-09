from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from typing import Optional, List, Mapping, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
class GeminiLLM(LLM):
    model: str = "gemini-2.0-flash"
    genai.configure(api_key=api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model_instance = genai.GenerativeModel(self.model)
        response = model_instance.generate_content([prompt])
        return response.text

    def generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        model_instance = genai.GenerativeModel(self.model)
        generations = []
        for prompt in prompts:
            response = model_instance.generate_content([prompt])
            generations.append([Generation(text=response.text)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "google-gemini"
