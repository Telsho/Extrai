import logging
from typing import Optional
from .generic_openai_client import GenericOpenAIClient


class OpenAIClient(GenericOpenAIClient):
    """
    LLM Client specifically for OpenAI models, inheriting from GenericOpenAIClient.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: Optional[float] = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the OpenAIClient.

        Args:
            api_key: The API key for OpenAI.
            model_name: The model name to use (e.g., "gpt-4o").
            base_url: The base URL for the OpenAI API.
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
