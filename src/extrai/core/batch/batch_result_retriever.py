import json
import logging

from extrai.core.analytics_collector import WorkflowAnalyticsCollector
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch_models import BatchJobContext
from extrai.core.cost_calculator import track_usage_from_response
from extrai.core.model_registry import ModelRegistry
from extrai.utils.llm_output_processing import process_and_validate_llm_output


class BatchResultRetriever:
    def __init__(
        self,
        model_registry: ModelRegistry,
        logger: logging.Logger,
        analytics_collector: WorkflowAnalyticsCollector | None = None,
    ):
        self.model_registry = model_registry
        self.logger = logger
        self.analytics_collector = analytics_collector

    async def retrieve_and_validate_results(
        self, context: BatchJobContext, client: BaseLLMClient
    ) -> tuple[list[dict], list[dict]]:
        results_content = await client.retrieve_batch_results(context.current_batch_id)

        # DEBUG: Log context info for diagnosis
        self.logger.debug(
            f"[BatchResultRetriever] current_model_index={context.config.current_model_index}, "
            f"hierarchical={context.config.hierarchical}, root_model={self.model_registry.root_model.__name__}"
        )

        # Determine the correct default model type based on current model index
        # This is critical for hierarchical extraction where we process models in order
        current_model_index = context.config.current_model_index
        if context.config.hierarchical and 0 <= current_model_index < len(
            self.model_registry.models
        ):
            default_model_type = self.model_registry.models[
                current_model_index
            ].__name__
            self.logger.debug(
                f"[BatchResultRetriever] Using hierarchical model type: {default_model_type}"
            )
        else:
            default_model_type = self.model_registry.root_model.__name__
            self.logger.debug(
                f"[BatchResultRetriever] Using root model type: {default_model_type}"
            )

        # Handle both string (JSONL) and list return types
        if isinstance(results_content, str):
            # Split by lines and filter empty lines
            result_lines = [
                l.strip() for l in results_content.strip().split("\n") if l.strip()
            ]
        else:
            result_lines = (
                results_content
                if isinstance(results_content, list)
                else [results_content]
            )

        validated_results = []
        validation_errors = []
        model_schema = self.model_registry.model_map

        for res in result_lines:
            try:
                # Extract the actual LLM content from the batch response wrapper
                # Batch responses are wrapped like: {"id": "...", "response": {"body": {"choices": [{"message": {"content": "..."}}]}}}
                raw_content = res
                if isinstance(res, str):
                    try:
                        parsed = json.loads(res)
                        # Try to extract content from batch response structure
                        try:
                            if self.analytics_collector:
                                track_usage_from_response(
                                    parsed,
                                    client,
                                    self.analytics_collector,
                                    context.current_batch_id,
                                )

                            extracted = client.extract_content_from_batch_response(
                                parsed
                            )
                            if extracted:
                                raw_content = extracted
                        except NotImplementedError:
                            # Client does not support extraction, proceed with parsed or raw
                            pass
                    except json.JSONDecodeError:
                        pass  # Use raw string as-is

                processed = process_and_validate_llm_output(
                    raw_content,
                    model_schema,
                    self.logger,
                    default_model_type=default_model_type,
                )
                validated_results.append({"revisions": processed})
            except Exception as e:
                self.logger.warning(
                    f"Validation failed for a batch result: {e}", exc_info=True
                )
                validation_errors.append({"original": res, "error": str(e)})

        return validated_results, validation_errors
