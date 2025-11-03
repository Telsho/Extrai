import logging
from typing import Optional
from .generic_openai_client import GenericOpenAIClient


class OllamaClient(GenericOpenAIClient):
    """
    LLM Client specifically for Ollama models, using an OpenAI-compatible interface.
    Inherits from GenericOpenAIClient to leverage common revision generation and validation logic.
    """

    def __init__(
        self,
        api_key: str = "ollama",  # Often not required, but good practice to have a default
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434/v1",
        temperature: Optional[float] = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the OllamaClient.

        Args:
            api_key: The API key for the Ollama service (if required).
            model_name: The specific Ollama model identifier.
            base_url: The base URL for the Ollama API (OpenAI-compatible endpoint).
            temperature: The sampling temperature for generation.
            logger: Logger.
        """
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            logger=logger,
        )
