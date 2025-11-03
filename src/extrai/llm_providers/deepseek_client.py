import logging
from typing import Optional
from .generic_openai_client import GenericOpenAIClient


class DeepSeekClient(GenericOpenAIClient):
    """
    LLM Client specifically for DeepSeek models, inheriting from GenericOpenAIClient.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        temperature: Optional[float] = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the DeepSeekClient.

        Args:
            api_key: The API key for DeepSeek.
            model_name: The model name to use (e.g., "deepseek-chat").
            base_url: The base URL for the DeepSeek API.
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
