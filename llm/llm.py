from typing import Type
from pydantic import BaseModel, ConfigDict


# Base class to manage LLM configuration.
class LLMSettings(BaseModel):
    # class instantiating the model
    _pyclass: Type = None

    # This is related to pydantic, because "model_*" attributes are protected.
    # We deactivate the protection because langchain relies on several "model_*" named attributes
    model_config = ConfigDict(protected_namespaces=())

    # instantiate an LLM from configuration
    @classmethod
    def get_llm_from_config(cls, config):
        if cls._pyclass is None:
            raise Exception(
                "Language model configuration class has self._pyclass==None. Should be a valid LLM class"
            )
        return cls._pyclass.default(**config)
