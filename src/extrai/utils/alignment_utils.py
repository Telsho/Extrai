from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher


def normalize_json_revisions(revisions: List[Any]) -> List[Any]:
    """
    Aligns arrays across revisions using similarity-based matching.
    Handles different structures and ensures consistent ordering.
    """
    if not revisions:
        return revisions

    # Check if all revisions are lists of dictionaries (entity arrays)
    if all(
        isinstance(rev, list) and rev and isinstance(rev[0], dict)
        for rev in revisions
        if rev
    ):
        return align_entity_arrays(revisions)

    # Check if revisions have a "results" wrapper
    if all(isinstance(rev, dict) and "results" in rev for rev in revisions):
        results_arrays = [rev["results"] for rev in revisions]
        aligned_results = align_entity_arrays(results_arrays)
        # Reconstruct with aligned results
        return [{"results": aligned} for aligned in aligned_results]

    # Otherwise return as-is (single object extractions)
    return revisions


def align_entity_arrays(
    arrays: List[List[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    """
    Aligns multiple arrays of entities so similar objects are in the same positions.
    Uses the first array as reference and matches objects based on similarity.
    """
    if not arrays or not any(arrays):
        return arrays

    # Validate all arrays have the same length
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        print(
            f"Warning: Arrays have different lengths {lengths}. Using minimum length."
        )
        min_length = min(lengths)
        arrays = [arr[:min_length] for arr in arrays]

    # Use first array as reference
    reference = arrays[0]
    aligned = [reference[:]]

    # Align each subsequent array to match the reference
    for arr in arrays[1:]:
        reordered = []
        used_indices = set()

        for ref_obj in reference:
            # Find best match in current array
            best_idx = find_best_match(ref_obj, arr, used_indices)
            reordered.append(arr[best_idx])
            used_indices.add(best_idx)

        aligned.append(reordered)

    return aligned


def find_best_match(
    target: Dict[str, Any], candidates: List[Dict[str, Any]], used_indices: set
) -> int:
    """
    Finds the index of the most similar object in candidates that hasn't been used.
    """
    best_idx = -1
    best_score = -1.0

    for idx, candidate in enumerate(candidates):
        if idx in used_indices:
            continue

        score = calculate_similarity(target, candidate)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def calculate_similarity(obj1: Dict[str, Any], obj2: Dict[str, Any]) -> float:
    """
    Calculates similarity score between two objects (0-1, higher is more similar).
    Handles different field types recursively.
    """
    if not isinstance(obj1, dict) or not isinstance(obj2, dict):
        return 1.0 if obj1 == obj2 else 0.0

    # Check for ID fields first (quick exact match)
    id1 = obj1.get("_temp_id") or obj1.get("id")
    id2 = obj2.get("_temp_id") or obj2.get("id")
    if id1 and id2 and str(id1) == str(id2):
        return 1.0

    # Get all unique fields
    all_fields = set(obj1.keys()) | set(obj2.keys())
    if not all_fields:
        return 1.0

    total_similarity = 0.0

    for field in all_fields:
        val1 = obj1.get(field)
        val2 = obj2.get(field)

        # If field missing in one object
        if field not in obj1 or field not in obj2:
            field_similarity = 0.0
        else:
            field_similarity = compare_values(val1, val2)

        total_similarity += field_similarity

    return total_similarity / len(all_fields)


def compare_values(val1: Any, val2: Any) -> float:
    """
    Compares two values and returns similarity score (0-1).
    """
    # Handle None
    if val1 is None and val2 is None:
        return 1.0
    if val1 is None or val2 is None:
        return 0.0

    # Prevent boolean vs number comparison (True == 1 is True in Python)
    if isinstance(val1, bool) != isinstance(val2, bool):
        return 0.0

    # Exact equality
    if val1 == val2:
        return 1.0

    # String comparison (fuzzy)
    if isinstance(val1, str) and isinstance(val2, str):
        # Case-insensitive comparison
        if val1.strip().lower() == val2.strip().lower():
            return 1.0
        # Fuzzy string matching
        return SequenceMatcher(None, val1, val2).ratio()

    # Numeric comparison
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        max_val = max(abs(val1), abs(val2), 1)
        return 1.0 - min(abs(val1 - val2) / max_val, 1.0)

    # List comparison (recursive)
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) == 0 or len(val2) == 0:
            return 0.0

        # Find best matches for each element
        similarities = []
        for item1 in val1:
            best_match = max(
                (compare_values(item1, item2) for item2 in val2), default=0.0
            )
            similarities.append(best_match)

        return sum(similarities) / len(similarities)

    # Dict comparison (recursive)
    if isinstance(val1, dict) and isinstance(val2, dict):
        return calculate_similarity(val1, val2)

    # Different types
    return 0.0
