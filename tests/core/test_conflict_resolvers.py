# tests/core/test_conflict_resolvers.py
import unittest
from extrai.core.conflict_resolvers import (
    levenshtein_similarity,
    SimilarityClusterResolver,
    prefer_most_common_resolver,
)


class TestConflictResolvers(unittest.TestCase):
    def test_levenshtein_similarity(self):
        # Identical
        self.assertEqual(levenshtein_similarity("abc", "abc"), 1.0)
        # Completely different
        self.assertEqual(levenshtein_similarity("abc", "def"), 0.0)
        # Partial
        # "apple" vs "aple" (1 deletion). Ratio > 0.8
        sim = levenshtein_similarity("apple", "aple")
        self.assertGreater(sim, 0.8)
        self.assertLess(sim, 1.0)

    def test_similarity_clustering_simple(self):
        resolver = SimilarityClusterResolver(similarity_threshold=0.6)
        # Cluster: "Christmas Party", "Xmas Party" (Maybe not close enough for Levenshtein, check ratio)
        # "Christmas" vs "Xmas" is distinct.
        # "Christmas Party" vs "Christmas PArty" is close.
        values = ["Christmas Party", "Christmas PArty", "War Zone"]
        # "Christmas Party" vs "Christmas PArty" -> ratio ~0.9
        # "War Zone" -> ratio ~0.1

        path = ("event",)
        result = resolver(path, values)
        # Should pick from the cluster ["Christmas Party", "Christmas PArty"]
        # prefer_most_common picks the first one in the cluster list usually
        self.assertIn(result, ["Christmas Party", "Christmas PArty"])
        self.assertNotEqual(result, "War Zone")

    def test_similarity_clustering_outlier(self):
        resolver = SimilarityClusterResolver(similarity_threshold=0.5)
        # 3 values. A and B close. C far.
        values = ["abcdefg", "abcdefh", "zzzzzzz"]
        result = resolver(("p",), values)
        self.assertIn(result, ["abcdefg", "abcdefh"])

    def test_similarity_clustering_weighted(self):
        resolver = SimilarityClusterResolver(similarity_threshold=0.5)
        values = ["A", "A'", "B"]
        # A and A' close. B far.
        # Weights: A (0.1), A' (0.1), B (0.8) -> B is huge weight but outlier in string space?
        # If B is outlier, it forms its own cluster [B].
        # Cluster [A, A'] has size 2.
        # Cluster [B] has size 1.
        # Logic says: prefer largest cluster?
        # Code says:
        #   if weights: best_cluster = max(clusters, key=cluster_weight)
        # Cluster [A, A'] weight = 0.2
        # Cluster [B] weight = 0.8
        # So B should win if we use weighted clustering logic!

        # NOTE: A and A' need to be similar. "abc" and "abd".
        values = ["abc", "abd", "zzz"]
        weights = [0.1, 0.1, 0.8]
        result = resolver(("p",), values, weights)
        self.assertEqual(result, "zzz")
        # This confirms that even if "abc" and "abd" are similar, "zzz" wins because of high trust (weight).

    def test_prefer_most_common_weighted(self):
        values = ["A", "B", "A"]
        weights = [0.1, 0.8, 0.05]
        # A total: 0.15
        # B total: 0.8
        # B wins
        result = prefer_most_common_resolver(("p",), values, weights)
        self.assertEqual(result, "B")


if __name__ == "__main__":
    unittest.main()
