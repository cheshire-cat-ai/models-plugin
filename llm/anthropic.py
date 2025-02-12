from typing import Type
from pydantic import ConfigDict

from langchain_anthropic import ChatAnthropic

from .llm import LLMSettings


class LLMAnthropicChatConfig(LLMSettings):
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 8192
    max_retries: int = 2

    _pyclass: Type = ChatAnthropic

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Anthropic",
            "description": "Configuration for Anthropic",
            "link": "https://www.anthropic.com/",
        }
    )
