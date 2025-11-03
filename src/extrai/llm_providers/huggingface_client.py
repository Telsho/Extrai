import logging
from typing import Optional
from .generic_openai_client import GenericOpenAIClient


class HuggingFaceClient(GenericOpenAIClient):
    """
    LLM Client for Hugging Face models, using an OpenAI-compatible interface.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        base_url: str = "https://api-inference.huggingface.co/v1/",
        temperature: Optional[float] = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the HuggingFaceClient.

        Args:
            api_key: The Hugging Face API token.
            model_name: The specific Hugging Face model identifier.
            base_url: The base URL for the Hugging Face Inference API (OpenAI-compatible endpoint).
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
