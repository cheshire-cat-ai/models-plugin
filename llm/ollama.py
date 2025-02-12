from typing import Any, Type
from pydantic import ConfigDict

from langchain_ollama import ChatOllama

from .llm import LLMSettings


class CustomOllama(ChatOllama):
    def __init__(self, **kwargs: Any) -> None:
        if kwargs["base_url"].endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        super().__init__(**kwargs)


class LLMOllamaConfig(LLMSettings):
    base_url: str
    model: str = "llama3"
    num_ctx: int = 2048
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temperature: float = 0.8

    _pyclass: Type = CustomOllama

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Ollama",
            "description": "Configuration for Ollama",
            "link": "https://ollama.ai/library",
        }
    )
