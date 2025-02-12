from typing import List
from cat.mad_hatter.decorators import tool, hook, plugin

from .llm.anthropic import LLMAnthropicChatConfig
from .llm.cohere import LLMCohereConfig
from .llm.gemini import LLMGeminiChatConfig
from .llm.hugging_face import LLMHuggingFaceEndpointConfig, LLMHuggingFaceTextGenInferenceConfig
from .llm.openai import (
    LLMOpenAIChatConfig,
    LLMOpenAIConfig,
    LLMOpenAICompatibleConfig,
    LLMAzureOpenAIConfig,
    LLMAzureChatOpenAIConfig,
)
from .llm.ollama import LLMOllamaConfig

from .embedder.embedder import (
    EmbedderOpenAICompatibleConfig,
    EmbedderOpenAIConfig,
    EmbedderAzureOpenAIConfig,
    EmbedderCohereConfig,
    EmbedderQdrantFastEmbedConfig,
    EmbedderGeminiChatConfig,
)

# TODO: Add settings to allow model management

llms = [
    LLMAnthropicChatConfig,
    LLMCohereConfig,
    LLMGeminiChatConfig,
    LLMHuggingFaceEndpointConfig,
    LLMHuggingFaceTextGenInferenceConfig,
    LLMOpenAIChatConfig,
    LLMOpenAIConfig,
    LLMOpenAICompatibleConfig,
    LLMAzureOpenAIConfig,
    LLMAzureChatOpenAIConfig,
    LLMOllamaConfig,
]

@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed = allowed + llms
    return allowed


embedders = [
    EmbedderOpenAICompatibleConfig,
    EmbedderOpenAIConfig,
    EmbedderAzureOpenAIConfig,
    EmbedderCohereConfig,
    EmbedderQdrantFastEmbedConfig,
    EmbedderGeminiChatConfig,
]


@hook
def factory_allowed_embedders(allowed, cat) -> List:
    allowed = allowed + embedders
    return allowed
