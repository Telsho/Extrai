import json
import logging
import os
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2 import service_account

try:
    from google import genai
except ImportError:
    genai = None

from .base_google_client import BaseGoogleGenAIClient


class VertexAIClient(BaseGoogleGenAIClient):
    """
    LLM Client specifically for Vertex AI models, inheriting from BaseGoogleGenAIClient.
    Supports either API Key or GCP Service Account JSON.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        service_account_json: str | dict | None = None,
        project_id: str | None = None,
        location: str = "global",
        temperature: float | None = 0.3,
        logger: logging.Logger | None = None,
        gcs_bucket_name: str | None = None,
    ):
        """
        Initializes the VertexAIClient.

        Args:
            model_name: The model name to use (e.g., "gemini-2.5-flash").
            api_key: The API key for Vertex/Gemini (if using direct API key).
            service_account_json: Path to the service account JSON file, or dict containing the credentials.
            project_id: GCP Project ID (can be inferred from service_account_json if provided).
            location: GCP region for Vertex AI (default: global).
            temperature: The sampling temperature for generation.
            logger: Logger.
            gcs_bucket_name: GCS bucket name for batch API usage.
        """
        credentials = None
        self.gcs_bucket_name = gcs_bucket_name

        if service_account_json:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            if isinstance(service_account_json, str):
                if os.path.exists(service_account_json):
                    credentials = service_account.Credentials.from_service_account_file(
                        service_account_json, scopes=scopes
                    )
                else:
                    try:
                        key_dict = json.loads(service_account_json)
                        credentials = (
                            service_account.Credentials.from_service_account_info(
                                key_dict, scopes=scopes
                            )
                        )
                    except json.JSONDecodeError:
                        raise ValueError(
                            "service_account_json must be a valid file path or valid JSON string."
                        )
            elif isinstance(service_account_json, dict):
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_json, scopes=scopes
                )

            if credentials:
                # Need an access token for the OpenAI-compatible endpoint
                credentials.refresh(Request())
                api_key = credentials.token

                # Infer project_id if not explicitly provided
                if not project_id and hasattr(credentials, "project_id"):
                    project_id = credentials.project_id

        if not project_id and credentials:
            project_id = getattr(credentials, "project_id", project_id)

        self._credentials = credentials
        self._project_id = project_id

        # Base URL for Vertex AI OpenAI-compatible endpoint
        if project_id:
            base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi"
        else:
            # Fallback to standard OpenAI compatible endpoint if project is unknown (e.g. standard Gemini API)
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        # api_key must be passed to GenericOpenAIClient. Fallback to dummy string if none.
        if not api_key:
            api_key = "dummy"

        # The OpenAI-compatible Vertex endpoint expects the model to have a publisher prefix
        # such as 'google/gemini-2.5-flash'. If no slash is present, add 'google/'.
        openai_model_name = model_name
        if "/" not in openai_model_name:
            openai_model_name = f"google/{openai_model_name}"

        super().__init__(
            api_key=api_key,
            model_name=openai_model_name,
            base_url=base_url,
            temperature=temperature,
            logger=logger,
        )
        # However, google-genai client expects the original model name, sometimes with 'models/'
        self.original_model_name = model_name

        if genai:
            client_args = {"vertexai": True}
            if credentials:
                client_args["credentials"] = credentials
                if project_id:
                    client_args["project"] = project_id
                client_args["location"] = location
            elif api_key and api_key != "dummy":
                client_args["api_key"] = api_key

            self.genai_client = genai.Client(**client_args)
        else:
            self.genai_client = None

    def create_inline_batch_job(
        self, src: list[Any], config: dict | None = None
    ) -> Any:
        if not self.genai_client:
            raise ImportError("google-genai package is required for this feature")

        model = getattr(self, "original_model_name", self.model_name)

        gcs_bucket = self.gcs_bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if gcs_bucket:
            import json
            import tempfile
            import uuid
            import os
            from google.cloud import storage

            job_id = str(uuid.uuid4())
            jsonl_lines = []
            for req in src:
                vertex_req = {"request": {"contents": req.get("contents", [])}}
                req_config = req.get("config", {})

                gen_config = {}
                if "response_mime_type" in req_config:
                    gen_config["responseMimeType"] = req_config["response_mime_type"]
                if "response_schema" in req_config:
                    gen_config["responseSchema"] = req_config["response_schema"]
                if "temperature" in req_config:
                    gen_config["temperature"] = req_config["temperature"]
                elif self.temperature is not None:
                    gen_config["temperature"] = self.temperature

                if gen_config:
                    vertex_req["request"]["generationConfig"] = gen_config

                if "system_instruction" in req_config:
                    vertex_req["request"]["systemInstruction"] = {
                        "role": "system",
                        "parts": [{"text": req_config["system_instruction"]}],
                    }
                jsonl_lines.append(json.dumps(vertex_req))

            jsonl_content = "\n".join(jsonl_lines)

            temp_prompt_dir = os.path.join(tempfile.gettempdir(), "temp_prompts")
            os.makedirs(temp_prompt_dir, exist_ok=True)
            prompt_file_path = os.path.join(temp_prompt_dir, f"{job_id}_prompt.jsonl")

            with open(prompt_file_path, "w") as f:
                f.write(jsonl_content + "\n")

            if hasattr(self, "_credentials") and self._credentials:
                storage_client = storage.Client(
                    credentials=self._credentials, project=self._project_id
                )
            else:
                storage_client = storage.Client()

            bucket = storage_client.bucket(gcs_bucket)
            input_blob_name = f"batch_inputs/{job_id}_prompt.jsonl"
            blob = bucket.blob(input_blob_name)
            blob.upload_from_filename(prompt_file_path)

            try:
                os.remove(prompt_file_path)
            except OSError:
                pass

            gcs_input_uri = f"gs://{gcs_bucket}/{input_blob_name}"
            gcs_output_uri = f"gs://{gcs_bucket}/batch_outputs/{job_id}/"

            if config is None:
                config = {}
            config["dest"] = gcs_output_uri
            config["display_name"] = f"batch_{job_id}"

            if self.logger:
                self.logger.info(
                    f"VertexAIClient: Uploaded batch inputs to {gcs_input_uri}. Output will be at {gcs_output_uri}"
                )

            return self.genai_client.batches.create(
                model=model,
                src=gcs_input_uri,
                config=config,
            )
        else:
            if self.logger:
                self.logger.warning(
                    "GCS_BUCKET_NAME not set, falling back to inline source"
                )

        return self.genai_client.batches.create(
            model=model,
            src=src,
            config=config,
        )

    async def retrieve_batch_results(self, batch_id: str) -> str:
        if not self.genai_client:
            return await super().retrieve_batch_results(batch_id)

        job = await self.retrieve_batch_job(batch_id)

        if hasattr(job, "dest") and hasattr(job.dest, "gcs_uri") and job.dest.gcs_uri:
            from google.cloud import storage
            import json

            if hasattr(self, "_credentials") and self._credentials:
                storage_client = storage.Client(
                    credentials=self._credentials, project=self._project_id
                )
            else:
                storage_client = storage.Client()

            gcs_uri = job.dest.gcs_uri
            bucket_name, blob_prefix = gcs_uri.replace("gs://", "").split("/", 1)
            bucket = storage_client.bucket(bucket_name)

            output_lines = []
            blobs = bucket.list_blobs(prefix=blob_prefix)
            for blob in blobs:
                if blob.name.endswith(".jsonl"):
                    jsonl_string = blob.download_as_string()
                    for line in jsonl_string.strip().split(b"\n"):
                        if not line.strip():
                            continue
                        response_data = json.loads(line)
                        content_text = ""
                        if "response" in response_data:
                            resp = response_data["response"]
                            if "text" in resp:
                                content_text = resp["text"]
                            elif "candidates" in resp and resp["candidates"]:
                                try:
                                    content_text = resp["candidates"][0]["content"][
                                        "parts"
                                    ][0]["text"]
                                except (KeyError, IndexError):
                                    pass

                        openai_resp = {
                            "id": "batch_req",
                            "response": {
                                "status_code": 200,
                                "body": {
                                    "choices": [{"message": {"content": content_text}}]
                                },
                            },
                        }

                        if (
                            "response" in response_data
                            and "usage_metadata" in response_data["response"]
                        ):
                            usage = response_data["response"]["usage_metadata"]
                            openai_resp["response"]["body"]["usage"] = {
                                "prompt_tokens": usage.get(
                                    "promptTokenCount",
                                    usage.get("prompt_token_count", 0),
                                ),
                                "completion_tokens": usage.get(
                                    "candidatesTokenCount",
                                    usage.get("candidates_token_count", 0),
                                ),
                                "total_tokens": usage.get(
                                    "totalTokenCount", usage.get("total_token_count", 0)
                                ),
                            }

                        output_lines.append(json.dumps(openai_resp))
            return "\n".join(output_lines)

        return await super().retrieve_batch_results(batch_id)
