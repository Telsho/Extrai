import logging
import openai
import json
import io
from typing import Optional, List, Dict, Any
from extrai.core.errors import LLMAPICallError
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.analytics_collector import WorkflowAnalyticsCollector


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

    async def _execute_llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> str:
        """
        Makes the actual API call to an OpenAI-compatible LLM.

        Args:
            system_prompt: The system prompt for the LLM.
            user_prompt: The user prompt for the LLM.
            analytics_collector: Optional analytics collector.

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

            if analytics_collector and hasattr(chat_completion, "usage") and chat_completion.usage:
                analytics_collector.record_llm_usage(
                    input_tokens=getattr(chat_completion.usage, "prompt_tokens", 0),
                    output_tokens=getattr(chat_completion.usage, "completion_tokens", 0),
                    model=self.model_name,
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

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Any,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
        **kwargs: Any
    ) -> Any:
        """
        Generates structured output using OpenAI's beta.chat.completions.parse.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=response_model,
                temperature=self.temperature
                if self.temperature is not None
                else openai.NOT_GIVEN,
                **kwargs
            )

            if analytics_collector and hasattr(completion, "usage") and completion.usage:
                analytics_collector.record_llm_usage(
                    input_tokens=getattr(completion.usage, "prompt_tokens", 0),
                    output_tokens=getattr(completion.usage, "completion_tokens", 0),
                    model=self.model_name,
                )

            message = completion.choices[0].message
            if message.refusal:
                raise LLMAPICallError(f"Model refused to generate structured output: {message.refusal}")
            
            return message.parsed

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

    async def create_batch_job(
        self,
        requests: List[Dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Creates a batch job for processing multiple requests.
        """
        try:
            # 1. Create JSONL content
            jsonl_lines = []
            for i, req in enumerate(requests):
                # Check if the request is already in batch format
                if "method" in req and "url" in req and "body" in req:
                    jsonl_lines.append(json.dumps(req))
                else:
                    # Construct batch request object
                    # Extract custom_id if present in the body, otherwise generate one
                    body = req.copy()
                    custom_id = body.pop("custom_id", f"req-{i}")

                    batch_req = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": endpoint,
                        "body": body,
                    }
                    jsonl_lines.append(json.dumps(batch_req))

            jsonl_content = "\n".join(jsonl_lines)

            # 2. Upload File
            # Create a bytes buffer for the file content
            file_obj = io.BytesIO(jsonl_content.encode("utf-8"))
            # The API expects a 'name' attribute for the file-like object, or a tuple (name, content)
            # file_obj.name = "batch_requests.jsonl"
            # Using tuple syntax for file upload: (filename, content, content_type)
            file_tuple = ("batch_requests.jsonl", file_obj, "application/json")

            batch_input_file = await self.client.files.create(
                file=file_tuple, purpose="batch"
            )

            # 3. Create Batch
            batch_job = await self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint=endpoint,
                completion_window=completion_window,
                metadata=metadata,
            )

            return batch_job

        except openai.APIError as e:
            raise LLMAPICallError(f"Batch creation failed: {e}") from e
        except Exception as e:
            raise LLMAPICallError(f"Unexpected error creating batch: {e}") from e

    async def retrieve_batch_job(self, batch_id: str) -> Any:
        """
        Retrieves the status and details of a batch job.
        """
        try:
            return await self.client.batches.retrieve(batch_id)
        except openai.APIError as e:
            raise LLMAPICallError(f"Failed to retrieve batch {batch_id}: {e}") from e

    async def list_batch_jobs(
        self, limit: int = 20, after: Optional[str] = None
    ) -> Any:
        """
        Lists batch jobs.
        """
        try:
            return await self.client.batches.list(limit=limit, after=after)
        except openai.APIError as e:
            raise LLMAPICallError(f"Failed to list batches: {e}") from e

    async def cancel_batch_job(self, batch_id: str) -> Any:
        """
        Cancels a batch job.
        """
        try:
            return await self.client.batches.cancel(batch_id)
        except openai.APIError as e:
            raise LLMAPICallError(f"Failed to cancel batch {batch_id}: {e}") from e

    async def retrieve_batch_results(self, file_id: str) -> str:
        """
        Retrieves the content of a batch output file.
        """
        try:
            content = await self.client.files.content(file_id)
            return content.text
        except openai.APIError as e:
            raise LLMAPICallError(
                f"Failed to retrieve file content {file_id}: {e}"
            ) from e

    def extract_content_from_batch_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Extracts content from OpenAI batch response item.
        """
        if "response" in response and "body" in response["response"]:
            body = response["response"]["body"]
            if "choices" in body and body["choices"]:
                return body["choices"][0]["message"]["content"]
        return None
