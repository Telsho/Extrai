import logging
from typing import Any

from .base_google_client import BaseGoogleGenAIClient

try:
    from google import genai
except ImportError:
    genai = None


class GeminiClient(BaseGoogleGenAIClient):
    """
    LLM Client specifically for Gemini models, inheriting from BaseGoogleGenAIClient.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature: float | None = 0.3,
        logger: logging.Logger | None = None,
    ):
        """
        Initializes the GeminiClient.

        Args:
            api_key: The API key for Gemini.
            model_name: The model name to use (e.g., "gemini-2.5-flash").
            base_url: The base URL for the Gemini API - openai compatible.
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
        if genai:
            self.genai_client = genai.Client(api_key=api_key)
        else:
            self.genai_client = None
