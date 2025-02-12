from typing import Type
from pydantic import ConfigDict

from langchain_cohere import ChatCohere

from .llm import LLMSettings


class LLMCohereConfig(LLMSettings):
    cohere_api_key: str
    model: str = "command"
    temperature: float = 0.7
    streaming: bool = True

    _pyclass: Type = ChatCohere

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Cohere",
            "description": "Configuration for Cohere language model",
            "link": "https://docs.cohere.com/docs/models",
        }
    )
