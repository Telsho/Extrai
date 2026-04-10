import asyncio
import json
import logging
from typing import Any

from extrai.core.base_llm_client import ProviderBatchStatus

from .generic_openai_client import GenericOpenAIClient

try:
    from google import genai
except ImportError:
    genai = None


def _resolve_refs(s, root=None):
    if root is None:
        root = s
    if isinstance(s, dict):
        s.pop("additionalProperties", None)
        if "$ref" in s:
            ref_path = s["$ref"]
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                import copy

                resolved_def = copy.deepcopy(root.get("$defs", {}).get(def_name, {}))
                resolved_def = _resolve_refs(resolved_def, root)
                new_schema = {}
                for k, v in s.items():
                    if k != "$ref":
                        if k.startswith("$"):
                            continue
                        new_schema[k] = _resolve_refs(v, root)
                for k, v in resolved_def.items():
                    if k.startswith("$"):
                        continue
                    new_schema[k] = v
                return new_schema
        new_schema = {}
        for k, v in s.items():
            if k.startswith("$"):
                continue
            new_schema[k] = _resolve_refs(v, root)
        return new_schema
    elif isinstance(s, list):
        return [_resolve_refs(item, root) for item in s]
    return s


class BaseGoogleGenAIClient(GenericOpenAIClient):
    """
    Base client for Google models (Gemini and Vertex AI) that share inline batching logic using google-genai.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        temperature: float | None = 0.3,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            logger=logger,
        )
        self.genai_client = None

    def create_inline_batch_job(
        self, src: list[Any], config: dict | None = None
    ) -> Any:
        """
        Creates an inline batch job using the Google GenAI SDK.
        """
        if not self.genai_client:
            raise ImportError("google-genai package is required for this feature")

        # Ensure model name has 'models/' prefix if not present, as often required by GenAI SDK
        model = self.model_name
        if not model.startswith("models/"):
            model = f"models/{model}"

        return self.genai_client.batches.create(
            model=model,
            src=src,
            config=config,
        )

    def get_inline_batch_job(self, name: str) -> Any:
        """
        Retrieves an inline batch job using the Google GenAI SDK.
        """
        if not self.genai_client:
            raise ImportError("google-genai package is required for this feature")
        return self.genai_client.batches.get(name=name)

    async def create_batch_job(
        self,
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        response_model: Any | None = None,
    ) -> Any:
        """
        Creates a batch job. If google-genai is available, uses inline batching.
        Overridden to support Google GenAI inline batching logic.
        """
        if not self.genai_client:
            return await super().create_batch_job(
                requests, endpoint, completion_window, metadata, response_model
            )

        google_requests = []
        for req in requests:
            body = req.get("body", {})
            messages = body.get("messages", [])

            system_instruction = None
            contents = []

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    contents.append({"role": "user", "parts": [{"text": content}]})

            config = {"response_mime_type": "application/json"}

            if response_model:
                try:
                    # Gemini does not support 'additionalProperties' in schema
                    import pydantic

                    if hasattr(response_model, "model_json_schema"):
                        schema = response_model.model_json_schema()
                    else:
                        schema = pydantic.TypeAdapter(response_model).json_schema()

                    schema = _resolve_refs(schema)
                    config["response_schema"] = schema

                    if self.logger:
                        self.logger.info(
                            "BaseGoogleGenAIClient: Enabled Structured Output with Sanitized Schema."
                        )
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to generate sanitized schema: {e}")
                    # Fallback
                    config["response_schema"] = response_model
            elif "response_format" in body:
                rf = body["response_format"]
                if rf.get("type") == "json_schema" and "json_schema" in rf:
                    # Pass the schema dict directly
                    raw_schema = rf["json_schema"].get("schema", {})
                    config["response_schema"] = _resolve_refs(raw_schema)
                    if self.logger:
                        self.logger.info(
                            "BaseGoogleGenAIClient: Enabled Structured Output (response_schema dict) for batch request."
                        )

            if system_instruction:
                config["system_instruction"] = system_instruction

            google_requests.append({"contents": contents, "config": config})

        if self.logger:
            self.logger.debug(
                f"Submitting {len(google_requests)} requests to inline batch."
            )
            if google_requests:
                self.logger.debug(
                    f"Sample request 0: {json.dumps(google_requests[0], default=str)}"
                )

        # Run sync call in thread to avoid blocking
        return await asyncio.to_thread(
            self.create_inline_batch_job, src=google_requests
        )

    async def retrieve_batch_job(self, batch_id: str) -> Any:
        """
        Retrieves batch job status.
        """
        if not self.genai_client:
            return await super().retrieve_batch_job(batch_id)
        return await asyncio.to_thread(self.get_inline_batch_job, name=batch_id)

    async def retrieve_batch_results(self, batch_id: str) -> str:
        """
        Retrieves batch results.
        For Google GenAI, results are inline in the job object.
        """
        if not self.genai_client:
            return await super().retrieve_batch_results(batch_id)

        job = await self.retrieve_batch_job(batch_id)

        if self.logger:
            self.logger.debug(f"Retrieved batch job: {batch_id}")
            try:
                self.logger.debug(
                    f"Job state/status: {getattr(job, 'state', getattr(job, 'status', 'unknown'))}"
                )
            except Exception:
                pass

        output_lines = []
        if hasattr(job, "dest") and hasattr(job.dest, "inlined_responses"):
            for resp in job.dest.inlined_responses:
                content_text = ""
                if resp.response:
                    if hasattr(resp.response, "text"):
                        content_text = resp.response.text
                    else:
                        content_text = str(resp.response)

                openai_resp = {
                    "id": "batch_req",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": content_text}}]},
                    },
                }

                # Extract usage if available
                if resp.response and hasattr(resp.response, "usage_metadata"):
                    usage = resp.response.usage_metadata
                    openai_resp["response"]["body"]["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                        "completion_tokens": getattr(
                            usage, "candidates_token_count", 0
                        ),
                        "total_tokens": getattr(usage, "total_token_count", 0),
                    }

                output_lines.append(json.dumps(openai_resp))

        return "\n".join(output_lines)

    async def get_batch_status(self, batch_id: str) -> "ProviderBatchStatus":
        """
        Retrieves batch job status and maps it to a standardized format.
        """
        if not self.genai_client:
            return await super().get_batch_status(batch_id)

        job = await asyncio.to_thread(self.get_inline_batch_job, name=batch_id)

        # Defensive access to state name, handling strings or enum-like objects
        state = getattr(job, "state", "unknown")
        state_name = getattr(state, "name", str(state))

        if state_name in ("JOB_STATE_SUCCEEDED", "SUCCEEDED"):
            return ProviderBatchStatus.COMPLETED
        elif state_name in (
            "JOB_STATE_FAILED",
            "JOB_STATE_EXPIRED",
            "FAILED",
            "EXPIRED",
        ):
            return ProviderBatchStatus.FAILED
        elif state_name in ("JOB_STATE_PENDING", "PENDING"):
            return ProviderBatchStatus.PENDING
        elif state_name in ("JOB_STATE_CANCELLED", "CANCELLED"):
            return ProviderBatchStatus.CANCELLED

        # All other states (RUNNING, PROCESSING, VALIDATING, UNSPECIFIED) are treated as PROCESSING
        return ProviderBatchStatus.PROCESSING
