from typing import Type
from pydantic import ConfigDict

from langchain_community.llms import (
    HuggingFaceTextGenInference,
    HuggingFaceEndpoint,
)

from .llm import LLMSettings


# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_endpoint.HuggingFaceEndpoint.html
class LLMHuggingFaceEndpointConfig(LLMSettings):
    endpoint_url: str
    huggingfacehub_api_token: str
    task: str = "text-generation"
    max_new_tokens: int = 512
    top_k: int = None
    top_p: float = 0.95
    temperature: float = 0.8
    return_full_text: bool = False
    _pyclass: Type = HuggingFaceEndpoint

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "HuggingFace Endpoint",
            "description": "Configuration for HuggingFace Endpoint language models",
            "link": "https://huggingface.co/inference-endpoints",
        }
    )


# https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_textgen_inference.html
class LLMHuggingFaceTextGenInferenceConfig(LLMSettings):
    inference_server_url: str
    max_new_tokens: int = 512
    top_k: int = 10
    top_p: float = 0.95
    typical_p: float = 0.95
    temperature: float = 0.01
    repetition_penalty: float = 1.03
    _pyclass: Type = HuggingFaceTextGenInference

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "HuggingFace TextGen Inference",
            "description": "Configuration for HuggingFace TextGen Inference",
            "link": "https://huggingface.co/text-generation-inference",
        }
    )
