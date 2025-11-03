# extrai/core/json_consensus.py
import logging
import math
from collections import Counter
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from extrai.utils.flattening_utils import (
    flatten_json,
    unflatten_json,
    Path,
    JSONValue,
    JSONObject,
    JSONArray,
    FlattenedJSON,
)

# Sentinel value to indicate that no consensus was reached for a path.
_NO_CONSENSUS = object()

# Define a type for a list of JSON revisions
JSONRevisions = List[Union[JSONObject, JSONArray]]

# Define conflict resolution strategies
ConflictResolutionStrategy = Callable[[Path, List[JSONValue]], Optional[JSONValue]]


def default_conflict_resolver(
    path: Path, values: List[JSONValue]
) -> Optional[JSONValue]:
    """
    Default conflict resolution: if no consensus, omit the field.
    (Or choose the most common, even if below threshold, or raise error - configurable)
    For this default, we'll omit.
    """
    # print(f"Conflict at path {path}: Values {values}. No consensus achieved. Omitting.")
    return None


def prefer_most_common_resolver(
    _path: Path, values: List[JSONValue]
) -> Optional[JSONValue]:
    """
    Conflict resolution: prefer the most common value even if it doesn't meet the threshold.
    If multiple values have the same highest frequency, it picks one arbitrarily (based on Counter behavior).
    """
    if not values:
        return None
    count = Counter(values)
    most_common_value, _ = count.most_common(1)[0]
    return most_common_value


class JSONConsensus:
    """
    Calculates a consensus JSON object from multiple JSON revisions.
    """

    def __init__(
        self,
        consensus_threshold: float = 0.5,
        conflict_resolver: Optional[
            ConflictResolutionStrategy
        ] = default_conflict_resolver,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the JSONConsensus processor.

        Args:
            consensus_threshold: The minimum proportion of revisions that must agree on a
                                 value for it to be included in the consensus.
                                 E.g., 0.5 means more than 50% agreement needed.
                                 A value of 0.0 would mean any single occurrence is enough (less useful).
                                 A value of 1.0 would mean unanimous agreement is required.
            conflict_resolver: A function to call when no value for a path meets the
                               consensus threshold. It takes the path and list of
                               conflicting values and returns a single value or None.
                               If None, a default resolver that omits the field is used.
            logger: An optional logger instance. If not provided, a default logger is created.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

        if not (0.0 < consensus_threshold <= 1.0):
            raise ValueError(
                "Extrai threshold must be between 0.0 (exclusive) and 1.0 (inclusive)."
            )
        self.consensus_threshold = consensus_threshold
        self.conflict_resolver = conflict_resolver

    def get_consensus(
        self, revisions: JSONRevisions
    ) -> Tuple[Union[JSONObject, JSONArray, JSONValue, None], Dict[str, Any]]:
        num_revisions = len(revisions)
        analytics = self._initialize_analytics(num_revisions)

        if not revisions:
            return {}, analytics

        path_to_values = self._aggregate_paths(revisions)
        analytics["unique_paths_considered"] = len(path_to_values)

        consensus_flat_json = self._build_consensus_json(
            path_to_values, num_revisions, analytics
        )
        analytics["paths_in_consensus_output"] = len(consensus_flat_json)

        final_consensus_object = self._build_final_object(
            consensus_flat_json, revisions
        )

        return final_consensus_object, analytics

    def _initialize_analytics(self, num_revisions: int) -> Dict[str, Any]:
        return {
            "revisions_processed": num_revisions,
            "unique_paths_considered": 0,
            "paths_in_consensus_output": 0,
            "paths_agreed_by_threshold": 0,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 0,
        }

    def _aggregate_paths(self, revisions: JSONRevisions) -> Dict[Path, List[JSONValue]]:
        path_to_values: Dict[Path, List[JSONValue]] = {}
        flattened_revisions = [flatten_json(rev) for rev in revisions]
        for flat_rev in flattened_revisions:
            for path, value in flat_rev.items():
                path_to_values.setdefault(path, []).append(value)
        return path_to_values

    def _build_consensus_json(
        self,
        path_to_values: Dict[Path, List[JSONValue]],
        num_revisions: int,
        analytics: Dict[str, Any],
    ) -> FlattenedJSON:
        consensus_flat_json: FlattenedJSON = {}
        for path, values in path_to_values.items():
            agreed_value = self._get_consensus_for_path(path, values, num_revisions)

            if agreed_value is not _NO_CONSENSUS:
                consensus_flat_json[path] = agreed_value
                analytics["paths_agreed_by_threshold"] += 1
            else:
                # This is a conflict. Record the disagreement details.
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

                # For '_temp_id' and '_type', always prefer the most common value.
                if path[-1] in ["_temp_id", "_type"]:
                    self.logger.debug(
                        f"Conflict at path '{'.'.join(map(str, path))}': "
                        f"Using most common value resolver for special attribute."
                    )
                    resolved_value = prefer_most_common_resolver(path, values)
                else:
                    self.logger.debug(
                        f"Conflict at path '{'.'.join(map(str, path))}': "
                        f"Invoking custom conflict resolver. Values: {values}"
                    )
                    resolved_value = self.conflict_resolver(path, values)

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
        return consensus_flat_json

    def _get_consensus_for_path(
        self, path: Path, values: List[JSONValue], num_revisions: int
    ) -> Union[JSONValue, object]:
        most_common_candidate = prefer_most_common_resolver(path, values)
        max_count = values.count(most_common_candidate)

        # Unanimity check
        is_unanimous = max_count == num_revisions
        if math.isclose(self.consensus_threshold, 1.0):
            return most_common_candidate if is_unanimous else _NO_CONSENSUS

        # Threshold check
        agreement_ratio = max_count / num_revisions
        if agreement_ratio > self.consensus_threshold:
            return most_common_candidate

        return _NO_CONSENSUS

    def _build_final_object(
        self, consensus_flat_json: FlattenedJSON, revisions: JSONRevisions
    ) -> Union[JSONObject, JSONArray, JSONValue, None]:
        if not consensus_flat_json and revisions:
            return [] if isinstance(revisions[0], list) else {}
        return unflatten_json(consensus_flat_json)
