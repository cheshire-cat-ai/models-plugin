from enum import Enum
from typing import Type

from pydantic import BaseModel, ConfigDict, Field
from langchain_community.embeddings import FakeEmbeddings, FastEmbedEmbeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastembed import TextEmbedding
from .custom_embeddings import CustomOpenAIEmbeddings
from langchain_cohere import CohereEmbeddings


# Base class to manage LLM configuration.
class EmbedderSettings(BaseModel):
    # class instantiating the embedder
    _pyclass: Type = None

    # This is related to pydantic, because "model_*" attributes are protected.
    # We deactivate the protection because langchain relies on several "model_*" named attributes
    model_config = ConfigDict(protected_namespaces=())

    # instantiate an Embedder from configuration
    @classmethod
    def get_embedder_from_config(cls, config):
        if cls._pyclass is None:
            raise Exception(
                "Embedder configuration class has self._pyclass==None. Should be a valid Embedder class"
            )
        return cls._pyclass.default(**config)


class EmbedderOpenAICompatibleConfig(EmbedderSettings):
    url: str
    _pyclass: Type = CustomOpenAIEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI-compatible API embedder",
            "description": "Configuration for OpenAI-compatible API embeddings",
            "link": "",
        }
    )


class EmbedderOpenAIConfig(EmbedderSettings):
    openai_api_key: str
    model: str = "text-embedding-ada-002"
    _pyclass: Type = OpenAIEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI Embedder",
            "description": "Configuration for OpenAI embeddings",
            "link": "https://platform.openai.com/docs/models/overview",
        }
    )


# https://python.langchain.com/en/latest/_modules/langchain/embeddings/openai.html#OpenAIEmbeddings
class EmbedderAzureOpenAIConfig(EmbedderSettings):
    openai_api_key: str
    model: str
    azure_endpoint: str
    openai_api_type: str = "azure"
    openai_api_version: str
    deployment: str

    _pyclass: Type = AzureOpenAIEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Azure OpenAI Embedder",
            "description": "Configuration for Azure OpenAI embeddings",
            "link": "https://azure.microsoft.com/en-us/products/ai-services/openai-service",
        }
    )


class EmbedderCohereConfig(EmbedderSettings):
    cohere_api_key: str
    model: str = "embed-multilingual-v2.0"
    _pyclass: Type = CohereEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Cohere Embedder",
            "description": "Configuration for Cohere embeddings",
            "link": "https://docs.cohere.com/docs/models",
        }
    )


# Enum for menu selection in the admin!
FastEmbedModels = Enum(
    "FastEmbedModels",
    {
        item["model"].replace("/", "_").replace("-", "_"): item["model"]
        for item in TextEmbedding.list_supported_models()
    },
)


class EmbedderQdrantFastEmbedConfig(EmbedderSettings):
    model_name: FastEmbedModels = Field(title="Model name", default="BAAI/bge-base-en")
    # Unknown behavior for values > 512.
    max_length: int = 512
    # as suggest on fastembed documentation, "passage" is the best option for documents.
    doc_embed_type: str = "passage"
    cache_dir: str = "cat/data/models/fast_embed"

    _pyclass: Type = FastEmbedEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Qdrant FastEmbed (Local)",
            "description": "Configuration for Qdrant FastEmbed",
            "link": "https://qdrant.github.io/fastembed/",
        }
    )


class EmbedderGeminiChatConfig(EmbedderSettings):
    """Configuration for Gemini Chat Embedder.

    This class contains the configuration for the Gemini Embedder.
    """

    google_api_key: str
    # Default model https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai
    model: str = "models/embedding-001"

    _pyclass: Type = GoogleGenerativeAIEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Google Gemini Embedder",
            "description": "Configuration for Gemini Embedder",
            "link": "https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings?hl=en",
        }
    )
