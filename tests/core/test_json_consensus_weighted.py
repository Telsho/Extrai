# tests/core/test_json_consensus_weighted.py
import unittest
from extrai.core.json_consensus import JSONConsensus
from extrai.core.conflict_resolvers import prefer_most_common_resolver

class TestJSONConsensusWeighted(unittest.TestCase):
    def test_weighted_consensus_outlier_rejection(self):
        # Scenario: 
        # Rev 1 and Rev 2 agree on 10 context fields.
        # Rev 3 disagrees on those 10.
        # On the 'target' field, all 3 disagree.
        # R1 -> X, R2 -> Z, R3 -> Y.
        
        # Standard voting (if order is R3, R1, R2) would pick Y because it appears first in the tie.
        # Weighted voting should identify R1 and R2 as a cluster (high mutual similarity) and R3 as outlier.
        # Thus R1 and R2 get high weights. R3 gets low weight.
        # Tie between X and Z (high weights) should win over Y (low weight).
        
        rev1 = {f"f{i}": "A" for i in range(10)}
        rev1["target"] = "X"
        
        rev2 = {f"f{i}": "A" for i in range(10)}
        rev2["target"] = "Z"
        
        rev3 = {f"f{i}": "B" for i in range(10)}
        rev3["target"] = "Y"
        
        # Put R3 first to bias standard voting towards Y
        revisions = [rev3, rev1, rev2]
        
        consensus = JSONConsensus(
            consensus_threshold=0.5,
            conflict_resolver=prefer_most_common_resolver
        ) 
        result, analytics = consensus.get_consensus(revisions)
        
        print(f"Weights: {analytics.get('revision_weights')}")
        print(f"Result Target: {result.get('target')}")
        
        # Check that target is NOT Y (the outlier's choice)
        self.assertNotEqual(result.get("target"), "Y")
        self.assertIn(result.get("target"), ["X", "Z"])
        
        # Check weights in analytics
        w = analytics.get("revision_weights")
        # R3 is index 0. Should have low weight.
        # R1 (idx 1) and R2 (idx 2) match on 10 fields.
        # R3 matches none.
        self.assertLess(w[0], w[1])
        self.assertLess(w[0], w[2])


    def test_weighted_consensus_with_strings(self):
        from extrai.core.conflict_resolvers import SimilarityClusterResolver
        
        # R1 and R2 match on many string fields.
        rev1 = {"message": "hello world"}
        rev2 = {"message": "hello people"}
        rev3 = {"message": "goodbye moon"}
        
        # Conflict field
        # R1: "Apple"
        # R2: "Banana"
        # R3: "Cherry"
        # All distinct. All semantically far.
        # But R1 and R2 have high weight. R3 low.
        
        rev1["fruit"] = "Apple"
        rev2["fruit"] = "Banana"
        rev3["fruit"] = "Cherry"
        
        revisions = [rev3, rev1, rev2]
        
        # Use SimilarityClusterResolver (which falls back to weighted most common if no clusters)
        resolver = SimilarityClusterResolver(similarity_threshold=0.8)
        consensus = JSONConsensus(consensus_threshold=1.0, conflict_resolver=resolver)
        
        result, analytics = consensus.get_consensus(revisions)
        
        # Should pick Apple or Banana. Definitely not Cherry.
        self.assertNotEqual(result.get("fruit"), "Cherry")
        self.assertIn(result.get("fruit"), ["Apple", "Banana"])
        
        # Check avg string similarity
        # On paths s0..s4: R1-R2 is 1.0. R1-R3 is low. R2-R3 low.
        # Avg for those paths: (1.0 + low + low)/3
        self.assertGreater(analytics.get("average_string_similarity"), 0.0)

if __name__ == "__main__":
    unittest.main()
