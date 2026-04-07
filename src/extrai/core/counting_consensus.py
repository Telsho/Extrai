import json
import logging
from typing import Any

from .extraction_config import ExtractionConfig
from ..utils.alignment_utils import align_entity_arrays, calculate_similarity


class CountingConsensus:
    """
    Implements a multi-revision consensus step specifically for the counting phase.
    Utilizes Levenshtein distance to evaluate similarity among returned string arrays,
    and falls back to a "resolver LLM" if there is too much discrepancy.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        llm_client,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)

    async def achieve_consensus(
        self,
        revisions: list[dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        target_json_schema: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Attempts to reach consensus across multiple counting revisions.
        If consensus fails, triggers a fallback LLM call to merge.

        Args:
            revisions: A list where each item is a parsed JSON response (dict) containing `counted_entities`.
            system_prompt: The original system prompt used for counting.
            user_prompt: The original user prompt used for counting.
            target_json_schema: The JSON schema to enforce on the fallback call (if using structured output).

        Returns:
            The final consensus list of counted entities.
        """
        if not revisions:
            return []

        if len(revisions) == 1:
            return revisions[0].get("counted_entities", [])

        # Extract just the entity lists
        entity_lists = [rev.get("counted_entities", []) for rev in revisions]

        # Ensure all lists actually exist and are lists
        entity_lists = [lst if isinstance(lst, list) else [] for lst in entity_lists]

        # Step 2a: Length Verification
        lengths = [len(lst) for lst in entity_lists]
        all_same_length = all(le == lengths[0] for le in lengths)

        consensus_reached = False
        best_list_idx = 0

        if all_same_length and lengths[0] > 0:
            # Step 2b: Levenshtein Distance Comparison
            # Align arrays using the longest as reference (since lengths are equal, the first is fine)
            aligned_arrays = align_entity_arrays(
                entity_lists, truncate_to_min_length=False
            )

            reference_array = aligned_arrays[0]
            avg_similarities = []

            for i in range(1, len(aligned_arrays)):
                current_array = aligned_arrays[i]
                sim_sum = 0.0

                for j in range(len(reference_array)):
                    sim = calculate_similarity(reference_array[j], current_array[j])
                    sim_sum += sim

                avg_sim = (
                    sim_sum / len(reference_array) if len(reference_array) > 0 else 1.0
                )
                avg_similarities.append(avg_sim)

            # If the average similarity across all matched pairs exceeds a threshold, consensus reached.
            # We can pick the reference list since it's most similar.
            overall_avg_sim = (
                sum(avg_similarities) / len(avg_similarities)
                if avg_similarities
                else 1.0
            )

            if overall_avg_sim >= self.config.counting_levenshtein_threshold:
                consensus_reached = True
                best_list_idx = 0
                self.logger.info(
                    f"Counting consensus reached with average similarity {overall_avg_sim:.2f}"
                )

        elif all_same_length and lengths[0] == 0:
            # All returned empty lists
            return []

        if consensus_reached:
            return entity_lists[best_list_idx]

        # Step 2c: Discrepancy & Fallback (LLM Resolution)
        self.logger.warning("Counting consensus failed. Triggering Merger LLM Call.")

        # We need to recreate the system prompt but with conflicting_revisions injected.
        # However, we only have the raw `system_prompt` string.
        # Actually, if we're inside the LLM call, we can append the revisions manually
        # to the existing system prompt.

        revisions_json = json.dumps(revisions, indent=2)
        merge_instructions = f"""

# MERGE REQUIRED:
Previous extraction attempts returned conflicting results. Here are the conflicting revisions:
{revisions_json}

Your task is to cross-reference these previous attempts with the text and provide the final, comprehensive, and correct list of entities, resolving any discrepancies.
"""
        new_system_prompt = system_prompt + merge_instructions

        # Ensure we use a client (could be a rotator)
        client = self.llm_client
        if isinstance(client, list):
            client = client[0]

        try:
            merged_result = await client.generate_and_validate_raw_json_output(
                system_prompt=new_system_prompt,
                user_prompt=user_prompt,
                target_json_schema=target_json_schema,
                num_revisions=1,
                max_validation_retries_per_revision=self.config.max_validation_retries_per_revision,
                attempt_unwrap=False,
            )

            if isinstance(merged_result, list) and merged_result:
                merged_result = merged_result[0]

            if isinstance(merged_result, dict) and "counted_entities" in merged_result:
                return merged_result.get("counted_entities", [])

        except Exception as e:
            self.logger.error(f"Fallback merger LLM call failed: {e}")
            # Fallback: return the longest list
            max_idx = lengths.index(max(lengths))
            return entity_lists[max_idx]

        # Ultimate fallback
        max_idx = lengths.index(max(lengths))
        return entity_lists[max_idx]
