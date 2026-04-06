# extrai/core/json_consensus.py
import logging
import math
from collections import Counter
from typing import Any

from extrai.core.conflict_resolvers import (
    ConflictResolutionStrategy,
    default_conflict_resolver,
    levenshtein_similarity,
    prefer_most_common_resolver,
)
from extrai.utils.flattening_utils import (
    FlattenedJSON,
    JSONArray,
    JSONObject,
    JSONValue,
    Path,
    flatten_json,
    unflatten_json,
)

# Sentinel value to indicate that no consensus was reached for a path.
_NO_CONSENSUS = object()

# Define a type for a list of JSON revisions
JSONRevisions = list[JSONObject | JSONArray]


class JSONConsensus:
    """
    Calculates a consensus JSON object from multiple JSON revisions.
    Supports weighted consensus based on global revision similarity.
    """

    def __init__(
        self,
        consensus_threshold: float = 0.5,
        conflict_resolver: ConflictResolutionStrategy
        | None = default_conflict_resolver,
        logger: logging.Logger | None = None,
    ):
        """
        Initializes the JSONConsensus processor.

        Args:
            consensus_threshold: The minimum proportion of revisions that must agree on a
                                 value for it to be included in the consensus.
            conflict_resolver: A function to call when no value for a path meets the
                               consensus threshold.
            logger: An optional logger instance.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

        if not (0.0 < consensus_threshold <= 1.0):
            raise ValueError(
                "Extrai threshold must be between 0.0 (exclusive) and 1.0 (inclusive)."
            )
        self.consensus_threshold = consensus_threshold
        self.conflict_resolver = conflict_resolver or default_conflict_resolver

    def get_consensus(
        self, revisions: JSONRevisions
    ) -> tuple[JSONObject | JSONArray | JSONValue | None, dict[str, Any]]:
        num_revisions = len(revisions)
        analytics = self._initialize_analytics(num_revisions)

        if not revisions:
            return {}, analytics

        # Calculate revision weights based on global similarity (Jaccard-like on flattened fields)
        revision_weights = self._calculate_revision_weights(revisions)
        analytics["revision_weights"] = revision_weights

        # Aggregate paths, keeping track of which revision provided which value
        path_to_values_with_indices = self._aggregate_paths(revisions)
        analytics["unique_paths_considered"] = len(path_to_values_with_indices)

        # Build consensus
        consensus_flat_json = self._build_consensus_json(
            path_to_values_with_indices, num_revisions, analytics, revision_weights
        )
        analytics["paths_in_consensus_output"] = len(consensus_flat_json)

        final_consensus_object = self._build_final_object(
            consensus_flat_json, revisions
        )

        if analytics["unique_paths_considered"] > 0:
            analytics["consensus_confidence_score"] = (
                analytics["paths_agreed_by_threshold"]
                / analytics["unique_paths_considered"]
            )

        return final_consensus_object, analytics

    def _initialize_analytics(self, num_revisions: int) -> dict[str, Any]:
        return {
            "revisions_processed": num_revisions,
            "unique_paths_considered": 0,
            "paths_in_consensus_output": 0,
            "paths_agreed_by_threshold": 0,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 0,
            "consensus_confidence_score": 0.0,
            "average_string_similarity": 0.0,  # Average Levenshtein ratio (1.0 = identical)
        }

    def _calculate_revision_weights(self, revisions: JSONRevisions) -> list[float]:
        """
        Calculates weights for each revision based on its similarity to other revisions.
        Revisions that are similar to others get higher weights (centrality).
        """
        n = len(revisions)
        if n <= 1:
            return [1.0] * n

        flat_revs = [flatten_json(r) for r in revisions]
        scores = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    scores[i][j] = 1.0
                    continue

                # Compare flat_revs[i] and flat_revs[j]
                keys_i = set(flat_revs[i].keys())
                keys_j = set(flat_revs[j].keys())
                common_keys = keys_i.intersection(keys_j)
                union_keys = keys_i.union(keys_j)

                if not union_keys:
                    similarity = 0.0
                else:
                    score_sum = 0.0
                    for k in common_keys:
                        val_i = flat_revs[i][k]
                        val_j = flat_revs[j][k]
                        if isinstance(val_i, str) and isinstance(val_j, str):
                            score_sum += levenshtein_similarity(val_i, val_j)
                        else:
                            score_sum += 1.0 if val_i == val_j else 0.0

                    similarity = score_sum / len(union_keys)

                scores[i][j] = similarity
                scores[j][i] = similarity

        # Weight for rev i = sum of similarities to others
        weights = []
        for i in range(n):
            w = sum(scores[i][j] for j in range(n) if i != j)
            weights.append(w)

        total = sum(weights)
        if total == 0:
            return [1.0 / n] * n

        return [w / total for w in weights]

    def _aggregate_paths(
        self, revisions: JSONRevisions
    ) -> dict[Path, list[tuple[JSONValue, int]]]:
        """
        Aggregates values for each path, preserving the source revision index.
        """
        path_to_values: dict[Path, list[tuple[JSONValue, int]]] = {}
        flattened_revisions = [flatten_json(rev) for rev in revisions]
        for idx, flat_rev in enumerate(flattened_revisions):
            for path, value in flat_rev.items():
                path_to_values.setdefault(path, []).append((value, idx))
        return path_to_values

    def _build_consensus_json(
        self,
        path_to_values: dict[Path, list[tuple[JSONValue, int]]],
        num_revisions: int,
        analytics: dict[str, Any],
        revision_weights: list[float],
    ) -> FlattenedJSON:
        consensus_flat_json: FlattenedJSON = {}

        total_string_sim = 0.0
        string_path_count = 0

        for path, values_with_indices in path_to_values.items():
            values = [v for v, i in values_with_indices]
            indices = [i for v, i in values_with_indices]
            current_weights = [revision_weights[i] for i in indices]

            # Analytics: Calculate average string similarity
            if len(values) > 1 and all(isinstance(v, str) for v in values):
                path_sim_sum = 0.0
                pair_count = 0
                for i in range(len(values)):
                    for j in range(i + 1, len(values)):
                        path_sim_sum += levenshtein_similarity(values[i], values[j])
                        pair_count += 1
                if pair_count > 0:
                    total_string_sim += path_sim_sum / pair_count
                    string_path_count += 1

            agreed_value = self._get_consensus_for_path(
                path, values, current_weights, num_revisions
            )

            if agreed_value is not _NO_CONSENSUS:
                consensus_flat_json[path] = agreed_value
                analytics["paths_agreed_by_threshold"] += 1
            else:
                # Conflict
                value_counts = Counter(values)
                disagreement_details = {
                    "path": ".".join(map(str, path)),
                    "values": [
                        {"value": v, "votes": c} for v, c in value_counts.items()
                    ],
                }
                analytics.setdefault("consensus_disagreements", []).append(
                    disagreement_details
                )

                # Special handling for _temp_id and _type
                if path[-1] in ["_temp_id", "_type"]:
                    self.logger.debug(
                        f"Conflict at path '{'.'.join(map(str, path))}': "
                        f"Using most common value resolver for special attribute."
                    )
                    resolved_value = prefer_most_common_resolver(
                        path, values, current_weights
                    )
                else:
                    self.logger.debug(
                        f"Conflict at path '{'.'.join(map(str, path))}': "
                        f"Invoking custom conflict resolver. Values: {values}"
                    )
                    resolved_value = self.conflict_resolver(
                        path, values, current_weights
                    )

                if resolved_value is not None:
                    self.logger.debug(
                        f"Path '{'.'.join(map(str, path))}' resolved by conflict resolver. "
                        f"Value set to: {resolved_value}"
                    )
                    consensus_flat_json[path] = resolved_value
                    analytics["paths_resolved_by_conflict_resolver"] += 1
                else:
                    self.logger.debug(
                        f"Path '{'.'.join(map(str, path))}' omitted as per conflict resolver."
                    )
                    analytics[
                        "paths_omitted_due_to_no_consensus_or_resolver_omission"
                    ] += 1

        if string_path_count > 0:
            analytics["average_string_similarity"] = (
                total_string_sim / string_path_count
            )

        return consensus_flat_json

    def _get_consensus_for_path(
        self,
        path: Path,
        values: list[JSONValue],
        weights: list[float],
        num_revisions: int,
    ) -> JSONValue | object:
        # Use weighted voting if weights provided
        most_common_candidate = prefer_most_common_resolver(path, values, weights)

        # Calculate agreement ratio based on weights if available, else count
        if weights and len(weights) == len(values):
            aggrement_ratio = sum(
                w for v, w in zip(values, weights) if v == most_common_candidate
            )
            max_count = values.count(most_common_candidate)
            is_unanimous = max_count == num_revisions
            if math.isclose(self.consensus_threshold, 1.0):
                return most_common_candidate if is_unanimous else _NO_CONSENSUS

            # Threshold check
            if aggrement_ratio > self.consensus_threshold:
                return most_common_candidate

            return _NO_CONSENSUS

        else:
            # Fallback to unweighted
            max_count = values.count(most_common_candidate)
            is_unanimous = max_count == num_revisions
            if math.isclose(self.consensus_threshold, 1.0):
                return most_common_candidate if is_unanimous else _NO_CONSENSUS

            agreement_ratio = max_count / num_revisions
            if agreement_ratio > self.consensus_threshold:
                return most_common_candidate

        return _NO_CONSENSUS

    def _build_final_object(
        self, consensus_flat_json: FlattenedJSON, revisions: JSONRevisions
    ) -> JSONObject | JSONArray | JSONValue | None:
        if not consensus_flat_json and revisions:
            return [] if isinstance(revisions[0], list) else {}
        return unflatten_json(consensus_flat_json)
