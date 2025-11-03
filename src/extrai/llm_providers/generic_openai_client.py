import logging
import openai
from typing import Optional
from extrai.core.errors import LLMAPICallError
from extrai.core.base_llm_client import BaseLLMClient


class GenericOpenAIClient(BaseLLMClient):
    """
    A generic LLM client that uses the openai library to interact with
    any OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        temperature: Optional[float] = 0.3,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the GenericOpenAIClient.

        Args:
            api_key: The API key for the LLM service.
            model_name: The model name to use.
            base_url: The base URL for the API.
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
        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def _execute_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Makes the actual API call to an OpenAI-compatible LLM.

        Args:
            system_prompt: The system prompt for the LLM.
            user_prompt: The user prompt for the LLM.

        Returns:
            The raw string content from the LLM response. Returns an empty string
            if the LLM response content is None.

        Raises:
            LLMAPICallError: If the API call fails or returns an error.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            chat_completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature
                if self.temperature is not None
                else openai.NOT_GIVEN,
            )

            response_content = chat_completion.choices[0].message.content
            return response_content if response_content is not None else ""

        except openai.APIError as e:
            error_message = str(e)
            if hasattr(e, "message") and e.message:
                error_message = e.message
            elif hasattr(e, "body") and e.body:
                if "message" in e.body:
                    error_message = e.body["message"]
                elif "error" in e.body and "message" in e.body["error"]:
                    error_message = e.body["error"]["message"]

            status_code = e.status_code if hasattr(e, "status_code") else "N/A"
            raise LLMAPICallError(
                f"API call failed. Status: {status_code}. Error: {error_message}"
            ) from e
        except Exception as e:
            raise LLMAPICallError(
                f"Unexpected error during API call: {type(e).__name__} - {str(e)}"
            ) from e
