import logging
from typing import Optional
from .generic_openai_client import GenericOpenAIClient


class GeminiClient(GenericOpenAIClient):
    """
    LLM Client specifically for Google Gemini models, using an OpenAI-compatible interface.
    Inherits from GenericOpenAIClient to leverage common revision generation and validation logic.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/",
        temperature: Optional[float] = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the GeminiClient.

        Args:
            api_key: The API key for the Gemini service.
            model_name: The specific Gemini model identifier.
            base_url: The base URL for the Gemini API (OpenAI-compatible endpoint).
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
