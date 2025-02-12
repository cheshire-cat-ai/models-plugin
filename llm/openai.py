from typing import Any, Type
from pydantic import ConfigDict

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAI
from langchain_openai import ChatOpenAI, OpenAI

from .llm import LLMSettings


class CustomOpenAI(ChatOpenAI):
    url: str

    def __init__(self, **kwargs):
        super().__init__(model_kwargs={}, base_url=kwargs["url"], **kwargs)


class LLMOpenAIChatConfig(LLMSettings):
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    streaming: bool = True
    _pyclass: Type = ChatOpenAI

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI ChatGPT",
            "description": "Chat model from OpenAI",
            "link": "https://platform.openai.com/docs/models/overview",
        }
    )


class LLMOpenAICompatibleConfig(LLMSettings):
    url: str
    temperature: float = 0.01
    model_name: str
    api_key: str
    streaming: bool = True
    _pyclass: Type = CustomOpenAI

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI-compatible API",
            "description": "Configuration for OpenAI-compatible APIs, e.g. llama-cpp-python server, text-generation-webui, OpenRouter, TinyLLM, TogetherAI and many others.",
            "link": "",
        }
    )


class LLMOpenAIConfig(LLMSettings):
    openai_api_key: str
    model_name: str = "gpt-3.5-turbo-instruct"
    temperature: float = 0.7
    streaming: bool = True
    _pyclass: Type = OpenAI

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI GPT-3",
            "description": "OpenAI GPT-3. More expensive but also more flexible than ChatGPT.",
            "link": "https://platform.openai.com/docs/models/overview",
        }
    )


# https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/reference#chat-completions
class LLMAzureChatOpenAIConfig(LLMSettings):
    openai_api_key: str
    model_name: str = "gpt-35-turbo"  # or gpt-4, use only chat models !
    azure_endpoint: str
    max_tokens: int = 2048
    openai_api_type: str = "azure"
    # Dont mix api versions https://github.com/hwchase17/langchain/issues/4775
    openai_api_version: str = "2023-05-15"

    azure_deployment: str
    streaming: bool = True
    _pyclass: Type = AzureChatOpenAI

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Azure OpenAI Chat Models",
            "description": "Chat model from Azure OpenAI",
            "link": "https://azure.microsoft.com/en-us/products/ai-services/openai-service",
        }
    )


# https://python.langchain.com/en/latest/modules/models/llms/integrations/azure_openai_example.html
class LLMAzureOpenAIConfig(LLMSettings):
    openai_api_key: str
    azure_endpoint: str
    max_tokens: int = 2048
    api_type: str = "azure"
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    # Current supported versions 2022-12-01, 2023-03-15-preview, 2023-05-15
    # Don't mix api versions: https://github.com/hwchase17/langchain/issues/4775
    api_version: str = "2023-05-15"
    azure_deployment: str = "gpt-35-turbo-instruct"
    model_name: str = "gpt-35-turbo-instruct"  # Use only completion models !
    streaming: bool = True
    _pyclass: Type = AzureOpenAI

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Azure OpenAI Completion models",
            "description": "Configuration for Cognitive Services Azure OpenAI",
            "link": "https://azure.microsoft.com/en-us/products/ai-services/openai-service",
        }
    )

