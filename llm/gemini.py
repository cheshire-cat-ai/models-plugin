from typing import Type
from pydantic import ConfigDict

from langchain_google_genai import ChatGoogleGenerativeAI

from .llm import LLMSettings


class LLMGeminiChatConfig(LLMSettings):
    """Configuration for the Gemini large language model (LLM).

    This class inherits from the `LLMSettings` class and provides default values for the following attributes:

    * `google_api_key`: The Google API key used to access the Google Natural Language Processing (NLP) API.
    * `model`: The name of the LLM model to use. In this case, it is set to "gemini".
    * `temperature`: The temperature of the model, which controls the creativity and variety of the generated responses.
    * `top_p`: The top-p truncation value, which controls the probability of the generated words.
    * `top_k`: The top-k truncation value, which controls the number of candidate words to consider during generation.
    * `max_output_tokens`: The maximum number of tokens to generate in a single response.

    The `LLMGeminiChatConfig` class is used to create an instance of the Gemini LLM model, which can be used to generate text in natural language.
    """

    google_api_key: str
    model: str = "gemini-1.5-pro-latest"
    temperature: float = 0.1
    top_p: int = 1
    top_k: int = 1
    max_output_tokens: int = 29000

    _pyclass: Type = ChatGoogleGenerativeAI

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Google Gemini",
            "description": "Configuration for Gemini",
            "link": "https://deepmind.google/technologies/gemini",
        }
    )
