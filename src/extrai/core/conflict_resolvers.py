# extrai/core/conflict_resolvers.py
from collections import Counter
from typing import List, Optional, Callable, Dict, Any, Union
from extrai.utils.flattening_utils import Path, JSONValue
from difflib import SequenceMatcher

# Define conflict resolution strategies
ConflictResolutionStrategy = Callable[[Path, List[JSONValue], Optional[List[float]]], Optional[JSONValue]]

def default_conflict_resolver(
    path: Path, values: List[JSONValue], weights: Optional[List[float]] = None
) -> Optional[JSONValue]:
    """
    Default conflict resolution: if no consensus, omit the field.
    """
    return None

def prefer_most_common_resolver(
    _path: Path, values: List[JSONValue], weights: Optional[List[float]] = None
) -> Optional[JSONValue]:
    """
    Conflict resolution: prefer the most common value.
    If weights are provided, prefers the value with the highest total weight.
    """
    if not values:
        return None
    
    if weights and len(weights) == len(values):
        # Weighted voting
        weighted_counts: Dict[Any, float] = {}
        # We need to handle unhashable types (like dicts/lists) if they appear in values
        # But JSONValue can be complex. Typically conflict resolution is on leaves (primitives).
        # Flattening utils usually produce primitives at leaves, but lists can be values if not recursed?
        # Assuming primitives for now (str, int, float, bool, None).
        
        for val, w in zip(values, weights):
            # If val is unhashable, we can't key it easily. 
            # Fallback to string repr or identity if needed, but for now assume hashable.
            try:
                weighted_counts[val] = weighted_counts.get(val, 0.0) + w
            except TypeError:
                # Unhashable type (e.g. list), skip optimization or use repr
                # For safety, let's just pick the first one if we can't count.
                # Or convert to tuple?
                # Let's rely on standard Counter behavior for fallback.
                pass
        
        if weighted_counts:
            # Pick value with max weight
            # Break ties by first occurrence (insertion order in weighted_counts)
            most_common_value = max(weighted_counts, key=weighted_counts.get)
            return most_common_value

    # Fallback to unweighted count
    # Note: Counter works with unhashable types? No.
    # If values contains unhashables, Counter(values) raises TypeError.
    # We should handle that, but original code assumed they work or didn't handle lists as values?
    # flattening_utils unflattening implies values are leaves.
    try:
        count = Counter(values)
        most_common_value, _ = count.most_common(1)[0]
        return most_common_value
    except TypeError:
        # Fallback for unhashable
        return values[0]

def levenshtein_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

class SimilarityClusterResolver:
    """
    Resolves conflicts by clustering values based on string similarity.
    Useful for filtering out outliers (e.g. "War" vs "Christmas", "Gifts").
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        scorer: Callable[[str, str], float] = levenshtein_similarity,
    ):
        self.similarity_threshold = similarity_threshold
        self.scorer = scorer

    def __call__(self, path: Path, values: List[JSONValue], weights: Optional[List[float]] = None) -> Optional[JSONValue]:
        if not values:
            return None

        # Only applicable if values are strings
        if not all(isinstance(v, str) for v in values):
            return prefer_most_common_resolver(path, values, weights)

        # 1. Compute pairwise similarities and build adjacency list
        n = len(values)
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                score = self.scorer(values[i], values[j])
                if score >= self.similarity_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        # 2. Find connected components (clusters)
        visited = set()
        clusters = []
        for i in range(n):
            if i not in visited:
                component = []
                stack = [i]
                visited.add(i)
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                clusters.append(component)

        if not clusters:
            return prefer_most_common_resolver(path, values, weights)

        # 3. Find the best cluster
        # If weights are provided, pick the cluster with the highest total weight.
        # Otherwise, pick the largest cluster.
        
        if weights and len(weights) == n:
            def cluster_weight(indices):
                return sum(weights[i] for i in indices)
            best_cluster_indices = max(clusters, key=cluster_weight)
        else:
            best_cluster_indices = max(clusters, key=len)

        # 4. Pick the representative from the best cluster
        cluster_values = [values[i] for i in best_cluster_indices]
        cluster_weights = [weights[i] for i in best_cluster_indices] if weights else None

        return prefer_most_common_resolver(path, cluster_values, cluster_weights)
