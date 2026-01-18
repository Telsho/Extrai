import logging
import uuid
import json
from typing import Optional, Dict, Any, List
from extrai.utils.rate_limiter import AsyncRateLimiter
from .generic_openai_client import GenericOpenAIClient
from extrai.core.errors import LLMAPICallError
from extrai.core.analytics_collector import WorkflowAnalyticsCollector


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
        requests_per_minute: int = 15,
        tokens_per_minute: int = 32000,
    ):
        """
        Initializes the GeminiClient.

        Args:
            api_key: The API key for the Gemini service.
            model_name: The specific Gemini model identifier.
            base_url: The base URL for the Gemini API (OpenAI-compatible endpoint).
            temperature: The sampling temperature for generation.
            logger: Logger.
            requests_per_minute: Maximum number of requests allowed per minute.
            tokens_per_minute: Maximum number of input tokens allowed per minute.
        """
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            logger=logger,
        )
        self.request_limiter = AsyncRateLimiter(max_capacity=requests_per_minute, period=60.0)
        self.token_limiter = AsyncRateLimiter(max_capacity=tokens_per_minute, period=60.0)
        self.logger = logger

    async def _execute_llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> str:
        """
        Executes the LLM call with rate limiting.
        """
        # Estimate token count (simple character heuristic)
        # 1 token ~= 4 chars
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
        # Minimum 1 token
        estimated_tokens = max(1, estimated_tokens)

        self.logger.warning("estimated tokens: " + str(estimated_tokens))
        # Acquire rate limits
        await self.request_limiter.acquire(1)
        await self.token_limiter.acquire(estimated_tokens)

        return await super()._execute_llm_call(
            system_prompt, user_prompt, analytics_collector=analytics_collector
        )

    def _sanitize_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures JSON schema compatibility with Gemini REST API by inlining $defs.
        Gemini REST API does not support $defs/$ref in schema payloads.
        This implements a dependency-free version of the 'jsonref' workaround.
        """
        import copy
        schema = copy.deepcopy(schema)
        defs = schema.pop("$defs", {}) or schema.pop("definitions", {})

        def _resolve(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref = node["$ref"].split("/")[-1]
                    if ref in defs:
                        return _resolve(defs[ref])
                return {k: _resolve(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [_resolve(x) for x in node]
            return node

        return _resolve(schema)

    async def create_batch_job(
        self,
        requests: List[Dict[str, Any]],
        endpoint: str = None,
        completion_window: str = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Creates a Gemini batch job using the native REST API (Inline Requests).
        """
        import httpx

        # Convert requests to Gemini 'contents' format
        gemini_requests = []
        for i, req in enumerate(requests):
            body = req.get("body", req)
            custom_id = req.get("custom_id", f"req-{i}")

            messages = body.get("messages", [])
            contents = []
            system_instruction = None

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    system_instruction = {"parts": [{"text": content}]}
                elif role == "user":
                    contents.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant":
                    contents.append({"role": "model", "parts": [{"text": content}]})
            
            # Construct the request object
            # Note: We need to ensure we use the correct model format
            # API expects model resource name in URL usually, but can also be in request?
            # Inline requests structure: { "request": { ... }, "metadata": ... }
            
            # Map configuration
            generation_config = {}
            if "temperature" in body:
                generation_config["temperature"] = body["temperature"]
            if "max_tokens" in body:
                generation_config["maxOutputTokens"] = body["max_tokens"]

            # Map OpenAI response_format to Gemini generationConfig
            response_format = body.get("response_format", {})
            if response_format.get("type") == "json_schema":
                generation_config["responseMimeType"] = "application/json"
                if "json_schema" in response_format and "schema" in response_format["json_schema"]:
                    raw_schema = response_format["json_schema"]["schema"]
                    generation_config["responseJsonSchema"] = self._sanitize_schema_for_gemini(raw_schema)
            elif response_format.get("type") == "json_object":
                generation_config["responseMimeType"] = "application/json"

            g_req_inner = {
                "contents": contents,
                "generationConfig": generation_config
            }
            if system_instruction:
                g_req_inner["system_instruction"] = system_instruction

            gemini_requests.append({
                "request": g_req_inner,
                "metadata": {"key": custom_id}
            })

        # Construct Payload
        payload = {
            "batch": {
                "input_config": {
                    "requests": {
                        "requests": gemini_requests
                    }
                }
            }
        }
        if metadata and "display_name" in metadata:
             payload["batch"]["display_name"] = metadata["display_name"]

        url = f"{self.base_url}models/{self.model_name}:batchGenerateContent?key={self.api_key}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code >= 400:
                raise LLMAPICallError(f"Gemini Batch Creation Failed: {resp.status_code} - {resp.text}")
            
            return self._wrap_batch_response(resp.json())

    async def retrieve_batch_job(self, batch_id: str) -> Any:
        """
        Retrieves batch status using Native REST API.
        """
        import httpx
        # batch_id is expected to be the full resource name e.g., "batches/12345"
        url = f"{self.base_url}{batch_id}?key={self.api_key}"
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                 raise LLMAPICallError(f"Gemini Batch Retrieve Failed: {resp.status_code} - {resp.text}")
            return self._wrap_batch_response(resp.json())

    async def cancel_batch_job(self, batch_id: str) -> Any:
        """
        Cancels batch job using Native REST API.
        """
        import httpx
        url = f"{self.base_url}{batch_id}:cancel?key={self.api_key}"
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url)
            if resp.status_code >= 400:
                 raise LLMAPICallError(f"Gemini Batch Cancel Failed: {resp.status_code} - {resp.text}")
            # Empty response usually on success? Or updated metadata.
            # We can re-fetch or just return true/empty
            return True

    def _wrap_batch_response(self, data: Dict[str, Any]) -> Any:
        class GeminiBatchJob:
            def __init__(self, data):
                self.id = data.get("name")  # "batches/..."
                # State might be at top level or inside metadata (depending on API version/endpoint)
                self.status = data.get("state")
                if not self.status and "metadata" in data:
                    self.status = data["metadata"].get("state")
                self.original_data = data

                # If finished, it might have results
                # Inline results structure?
                # The docs say: response.inlinedResponses
                # We can try to extract output_file_id if it exists, or handle inline.
                # OpenAI interface expects output_file_id for retrieve_batch_results.
                # If inline, we can't provide a file ID. 
                # We'll need retrieve_batch_results to handle the batch_id as file_id for inline.
                
        return GeminiBatchJob(data)

    async def retrieve_batch_results(self, file_id: str) -> str:
        """
        Retrieves batch results. 
        For Gemini Inline, 'file_id' should be the batch ID.
        """
        # If the batch job had 'responsesFile', we download it.
        # If it had 'inlinedResponses', we format it as JSONL.
        
        # We need to fetch the batch first to see which one it is (or assume we have the object)
        # But this method usually takes just an ID.
        # So we fetch the batch.
        
        batch = await self.retrieve_batch_job(file_id)
        data = batch.original_data
        
        # Check for inline responses
        # Structure: data.get("response", {}).get("inlinedResponses", [])
        # Actually docs say: batch_job.dest.inlined_responses (SDK) or .response.inlinedResponses (REST)
        
        # REST: .response.inlinedResponses
        response_section = data.get("response", {}) # Not to be confused with 'responses'
        # Wait, the example output JSON says:
        # "response": { "inlinedResponses": [ ... ] } OR "response": { "responsesFile": "..." }
        
        inlined = response_section.get("inlinedResponses")
        if inlined:
             # Handle case where inlined might be a dict (unexpected but observed)
             if isinstance(inlined, dict):
                 # If it's a dict, maybe the list is nested or it's a map?
                 self.logger.warning(f"inlinedResponses is a dict, keys: {list(inlined.keys())}")
                 # Try to find the actual list
                 if "inlinedResponses" in inlined:
                     inlined = inlined["inlinedResponses"]
                 elif "responses" in inlined:
                     inlined = inlined["responses"]
                 elif "results" in inlined:
                     inlined = inlined["results"]
                 else:
                     # Fallback: treat values as the list if they look like items
                     inlined = list(inlined.values())

             # Convert to JSONL string to match OpenAI format
             lines = []
             for item in inlined:
                 # item has 'response' or 'error' and 'requestKey' (if we used metadata.key)
                 # We should map back to OpenAI-like format if possible
                 if isinstance(item, str):
                      self.logger.warning(f"Unexpected string item in inlinedResponses: {item}")
                      continue
                 lines.append(json.dumps(item))
             return "\n".join(lines)
             
        file_name = response_section.get("responsesFile")
        if file_name:
             # Download file
             # url: https://generativelanguage.googleapis.com/download/v1beta/$responses_file_name:download?alt=media
             import httpx
             url = f"https://generativelanguage.googleapis.com/download/v1beta/{file_name}:download?alt=media&key={self.api_key}"
             async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                if resp.status_code >= 400:
                    raise LLMAPICallError(f"Gemini Result Download Failed: {resp.status_code}")
                return resp.text
                
        raise LLMAPICallError("No results found in batch (or batch not complete).")

    async def list_batch_jobs(self, limit: int = 20, after: Optional[str] = None) -> Any:
        import httpx
        url = f"{self.base_url}batches?key={self.api_key}&pageSize={limit}"
        if after:
            url += f"&pageToken={after}"
            
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                 raise LLMAPICallError(f"Gemini List Batches Failed: {resp.text}")
            
            data = resp.json()
            # Wrap list?
            return data

    def extract_content_from_batch_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Extracts content from Gemini batch response item.
        """
        if "error" in response:
            self.logger.error(f"Batch item contains error: {response['error']}")
            return None

        if "response" in response and "candidates" in response["response"]:
            candidates = response["response"]["candidates"]
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts:
                    return parts[0].get("text")
        return None
