# tests/test_json_consensus.py
import unittest

# If they are in extrai.core and extrai.utils
from extrai.core.json_consensus import (
    JSONConsensus,
    default_conflict_resolver,
    prefer_most_common_resolver,
)
from extrai.utils.flattening_utils import (
    JSONArray,
)  # For type hints if needed


class TestJSONConsensus(unittest.TestCase):
    def test_empty_revisions(self):
        consensus_processor = JSONConsensus()
        revisions: JSONArray = []
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, {})
        self.assertEqual(analytics["revisions_processed"], 0)
        self.assertEqual(analytics["unique_paths_considered"], 0)

    def test_single_revision(self):
        consensus_processor = JSONConsensus()
        rev1 = {"a": 1, "b": "hello"}
        revisions = [rev1]
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, rev1)
        self.assertEqual(analytics["revisions_processed"], 1)
        self.assertEqual(analytics["unique_paths_considered"], 2)  # a, b
        self.assertEqual(analytics["paths_agreed_by_threshold"], 2)
        self.assertEqual(analytics["paths_in_consensus_output"], 2)

    def test_perfect_agreement_dict(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.5)  # More than 50%
        rev1 = {"name": "Alice", "age": 30, "city": "New York"}
        rev2 = {"name": "Alice", "age": 30, "city": "New York"}
        rev3 = {"name": "Alice", "age": 30, "city": "New York"}
        revisions = [rev1, rev2, rev3]
        expected = {"name": "Alice", "age": 30, "city": "New York"}
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(analytics["revisions_processed"], 3)
        self.assertEqual(analytics["unique_paths_considered"], 3)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 3)
        self.assertEqual(analytics["paths_in_consensus_output"], 3)

    def test_perfect_agreement_list_of_primitives(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.5)
        rev1 = [1, 2, 3]
        rev2 = [1, 2, 3]
        revisions = [rev1, rev2]
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, [1, 2, 3])
        self.assertEqual(analytics["revisions_processed"], 2)
        self.assertEqual(analytics["unique_paths_considered"], 3)  # (0,), (1,), (2,)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 3)
        self.assertEqual(analytics["paths_in_consensus_output"], 3)

    def test_simple_majority_agreement_dict(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.5)  # Needs > 50%
        rev1 = {"name": "Alice", "age": 30}
        rev2 = {"name": "Alice", "age": 31}
        rev3 = {"name": "Alice", "age": 30}
        revisions = [rev1, rev2, rev3]  # Alice: 3/3, age 30: 2/3, age 31: 1/3
        expected = {"name": "Alice", "age": 30}  # age 30 has 2/3 > 0.5
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 2)  # name, age
        self.assertEqual(analytics["paths_in_consensus_output"], 2)
        self.assertEqual(
            analytics["unique_paths_considered"], 2
        )  # name, age (value for age is a list of [30,31,30])

    def test_no_clear_majority_default_resolver_dict(self):
        # Default resolver omits conflicting fields
        consensus_processor = JSONConsensus(
            consensus_threshold=0.5, conflict_resolver=default_conflict_resolver
        )
        rev1 = {"status": "active", "count": 1}
        rev2 = {"status": "inactive", "count": 2}
        rev3 = {"status": "pending", "count": 3}
        revisions = [rev1, rev2, rev3]  # status: 1/3 each, count: 1/3 each. None > 0.5
        expected = {}  # All fields omitted
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 0)
        self.assertEqual(
            analytics["paths_resolved_by_conflict_resolver"], 0
        )  # Default resolver returns None
        self.assertEqual(
            analytics["paths_omitted_due_to_no_consensus_or_resolver_omission"], 2
        )
        self.assertEqual(analytics["paths_in_consensus_output"], 0)

    def test_no_clear_majority_prefer_most_common_resolver_dict(self):
        consensus_processor = JSONConsensus(
            consensus_threshold=0.6,  # High threshold, likely conflict
            conflict_resolver=prefer_most_common_resolver,
        )
        rev1 = {"color": "red", "value": 10}
        rev2 = {"color": "blue", "value": 10}
        rev3 = {"color": "red", "value": 20}
        revisions = [rev1, rev2, rev3]
        # color: red (2/3), blue (1/3). value: 10 (2/3), 20 (1/3)
        # Threshold 0.6: 2/3 = 0.666... > 0.6. So red and 10 should pass.
        expected = {"color": "red", "value": 10}
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(
            analytics["paths_agreed_by_threshold"], 2
        )  # Both color and value met threshold
        self.assertEqual(analytics["paths_resolved_by_conflict_resolver"], 0)
        self.assertEqual(analytics["paths_in_consensus_output"], 2)

        # Test when most common does NOT meet threshold
        consensus_processor_strict_threshold = JSONConsensus(
            consensus_threshold=0.7,  # Requires 3/3 for 3 revisions (0.7*3=2.1, so need 3)
            conflict_resolver=prefer_most_common_resolver,
        )
        # color: red (2/3), blue (1/3) -> red is most common but 2/3 is not > 0.7
        # value: 10 (2/3), 20 (1/3) -> 10 is most common but 2/3 is not > 0.7
        # prefer_most_common_resolver will still pick them.
        expected_conflict_resolved = {"color": "red", "value": 10}
        consensus_obj_strict, analytics_strict = (
            consensus_processor_strict_threshold.get_consensus(revisions)
        )
        self.assertEqual(consensus_obj_strict, expected_conflict_resolved)
        self.assertEqual(
            analytics_strict["paths_agreed_by_threshold"], 0
        )  # None met the 0.7 threshold
        self.assertEqual(
            analytics_strict["paths_resolved_by_conflict_resolver"], 2
        )  # Both resolved
        self.assertEqual(analytics_strict["paths_in_consensus_output"], 2)

    def test_nested_structure_agreement(self):  # This was the original nested test
        consensus_processor = JSONConsensus(consensus_threshold=0.5)
        rev1 = {"user": {"name": "Bob", "details": {"id": 101, "verified": True}}}
        rev2 = {"user": {"name": "Bob", "details": {"id": 101, "verified": True}}}
        rev3 = {
            "user": {"name": "Bob", "details": {"id": 101, "verified": False}}
        }  # verified differs
        revisions = [rev1, rev2, rev3]
        # name: 3/3, id: 3/3, verified True: 2/3, verified False: 1/3
        expected = {"user": {"name": "Bob", "details": {"id": 101, "verified": True}}}
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(
            analytics["unique_paths_considered"], 3
        )  # user.name, user.details.id, user.details.verified
        self.assertEqual(analytics["paths_agreed_by_threshold"], 3)
        self.assertEqual(analytics["paths_in_consensus_output"], 3)

    def test_deeply_nested_json_object_consensus(self):  # NEW TEST
        consensus_processor = JSONConsensus(
            consensus_threshold=0.5
        )  # Needs > 50% agreement

        rev1 = {
            "document_id": "doc1",
            "metadata": {
                "source": "web",
                "timestamp": "2023-01-01T10:00:00Z",
                "author": {"name": "Alice", "role": "editor"},
            },
            "content": {
                "title": "Journey to the Center",
                "chapters": [
                    {"id": 1, "name": "The Beginning", "pages": 20},
                    {
                        "id": 2,
                        "name": "The Discovery",
                        "pages": 25,
                        "keywords": ["cave", "map"],
                    },
                ],
                "summary": "A grand adventure.",
            },
            "status": "final",
        }
        rev2 = {  # Differs in timestamp, author role, chapter 2 pages, chapter 2 keywords, summary, status
            "document_id": "doc1",
            "metadata": {
                "source": "web",
                "timestamp": "2023-01-01T10:05:00Z",  # Different
                "author": {"name": "Alice", "role": "writer"},  # Different role
            },
            "content": {
                "title": "Journey to the Center",
                "chapters": [
                    {"id": 1, "name": "The Beginning", "pages": 20},
                    {
                        "id": 2,
                        "name": "The Discovery",
                        "pages": 30,
                        "keywords": ["cave", "secret"],
                    },  # Diff pages, one keyword diff
                ],
                "summary": "An epic quest.",  # Different
            },
            "status": "draft",  # Different
        }
        rev3 = {  # Matches rev1 on most differing fields
            "document_id": "doc1",
            "metadata": {
                "source": "web",
                "timestamp": "2023-01-01T10:00:00Z",
                "author": {"name": "Alice", "role": "editor"},
            },
            "content": {
                "title": "Journey to the Center",
                "chapters": [
                    {"id": 1, "name": "The Beginning", "pages": 20},
                    {
                        "id": 2,
                        "name": "The Discovery",
                        "pages": 25,
                        "keywords": ["cave", "map"],
                    },
                ],
                "summary": "A grand adventure.",
            },
            "status": "final",
        }

        revisions = [rev1, rev2, rev3]

        expected_consensus = {
            "document_id": "doc1",  # 3/3
            "metadata": {
                "source": "web",  # 3/3
                "timestamp": "2023-01-01T10:00:00Z",  # 2/3 (rev1, rev3)
                "author": {
                    "name": "Alice",
                    "role": "editor",
                },  # name 3/3, role 2/3 (rev1, rev3)
            },
            "content": {
                "title": "Journey to the Center",  # 3/3
                "chapters": [
                    {"id": 1, "name": "The Beginning", "pages": 20},  # All 3/3
                    {
                        "id": 2,
                        "name": "The Discovery",  # id, name 3/3
                        "pages": 25,  # 2/3 (rev1, rev3)
                        "keywords": [
                            "cave",
                            "map",
                        ],  # keywords[0] "cave" 3/3, keywords[1] "map" 2/3 (rev1,rev3)
                    },
                ],
                "summary": "A grand adventure.",  # 2/3 (rev1, rev3)
            },
            "status": "final",  # 2/3 (rev1, rev3)
        }
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected_consensus)
        # Example analytics checks (can be more detailed)
        self.assertEqual(analytics["revisions_processed"], 3)
        self.assertGreater(
            analytics["unique_paths_considered"], 10
        )  # Count actual paths if needed
        self.assertGreater(analytics["paths_agreed_by_threshold"], 5)
        self.assertEqual(
            analytics["paths_in_consensus_output"],
            analytics["paths_agreed_by_threshold"],
        )  # Assuming no conflict resolver used here that adds new paths

    def test_lists_in_json(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.5)
        rev1 = {"id": 1, "tags": ["A", "B"], "scores": [10, 20]}
        rev2 = {"id": 1, "tags": ["A", "C"], "scores": [10, 20]}
        rev3 = {"id": 1, "tags": ["A", "B"], "scores": [10, 30]}
        revisions = [rev1, rev2, rev3]
        # id: 1 (3/3)
        # tags[0]: "A" (3/3)
        # tags[1]: "B" (2/3), "C" (1/3) -> "B" wins
        # scores[0]: 10 (3/3)
        # scores[1]: 20 (2/3), 30 (1/3) -> 20 wins
        expected = {"id": 1, "tags": ["A", "B"], "scores": [10, 20]}
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(
            analytics["unique_paths_considered"], 5
        )  # id, tags.0, tags.1, scores.0, scores.1
        self.assertEqual(analytics["paths_agreed_by_threshold"], 5)
        self.assertEqual(analytics["paths_in_consensus_output"], 5)

    def test_different_structures_partial_agreement(self):
        # Default resolver will omit fields not meeting consensus or not present enough times
        consensus_processor = JSONConsensus(consensus_threshold=0.5)
        rev1 = {"item": "apple", "price": 1.0, "info": {"available": True}}
        rev2 = {"item": "apple", "price": 1.2}  # No 'info'
        rev3 = {
            "item": "apple",
            "price": 1.0,
            "info": {"available": True, "organic": False},
        }
        revisions = [rev1, rev2, rev3]
        # item: apple (3/3)
        # price: 1.0 (2/3), 1.2 (1/3) -> 1.0 wins
        # info.available: True (2/3 from rev1, rev3) -> True wins
        # info.organic: False (1/3 from rev3) -> Fails threshold, omitted by default resolver
        # The path ('info',) itself exists in 2/3 revisions.
        # The unflattening should reconstruct 'info' if any sub-paths under it make it.
        expected = {"item": "apple", "price": 1.0, "info": {"available": True}}
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(
            analytics["unique_paths_considered"], 4
        )  # item, price, info.available, info.organic
        self.assertEqual(
            analytics["paths_agreed_by_threshold"], 3
        )  # item, price, info.available
        self.assertEqual(
            analytics["paths_omitted_due_to_no_consensus_or_resolver_omission"], 1
        )  # info.organic
        self.assertEqual(analytics["paths_in_consensus_output"], 3)

    def test_threshold_of_one(self):
        # Unanimous agreement needed
        consensus_processor = JSONConsensus(consensus_threshold=1.0)
        rev1 = {"a": 1, "b": 2}
        rev2 = {"a": 1, "b": 3}  # 'b' differs
        revisions = [rev1, rev2]
        expected = {"a": 1}  # 'b' is omitted due to conflict with 1.0 threshold
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 1)  # only 'a'
        self.assertEqual(
            analytics["paths_omitted_due_to_no_consensus_or_resolver_omission"], 1
        )  # 'b' omitted

        # Also test with a threshold very close to 1.0 to cover math.isclose
        consensus_processor_close = JSONConsensus(consensus_threshold=1.0 - 1e-9)
        consensus_obj_close, _ = consensus_processor_close.get_consensus(revisions)
        self.assertEqual(consensus_obj_close, expected)

        rev3 = {"a": 1, "b": 2}
        rev4 = {"a": 1, "b": 2}
        revisions_unanimous = [rev3, rev4]
        expected_unanimous = {"a": 1, "b": 2}
        consensus_obj_u, analytics_u = consensus_processor.get_consensus(
            revisions_unanimous
        )
        self.assertEqual(consensus_obj_u, expected_unanimous)
        self.assertEqual(analytics_u["paths_agreed_by_threshold"], 2)

    def test_list_of_objects_consensus(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.5)
        rev1 = [{"id": 1, "val": "x"}, {"id": 2, "val": "y"}]
        rev2 = [{"id": 1, "val": "x"}, {"id": 2, "val": "z"}]  # val for id 2 differs
        rev3 = [
            {"id": 1, "val": "x"},
            {"id": 2, "val": "y"},
        ]  # val for id 2 matches rev1
        revisions = [rev1, rev2, rev3]
        # (0, 'id'): 1 (3/3)
        # (0, 'val'): 'x' (3/3)
        # (1, 'id'): 2 (3/3)
        # (1, 'val'): 'y' (2/3), 'z' (1/3) -> 'y' wins
        expected = [{"id": 1, "val": "x"}, {"id": 2, "val": "y"}]
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(
            analytics["unique_paths_considered"], 4
        )  # (0,id), (0,val), (1,id), (1,val)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 4)

    def test_list_revisions_empty_consensus(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.9)  # High threshold
        rev1 = [{"id": 1}]
        rev2 = [{"id": 2}]
        revisions = [rev1, rev2]  # No consensus on any path
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, [])  # Expect an empty list
        self.assertEqual(analytics["paths_in_consensus_output"], 0)
        self.assertEqual(
            analytics["paths_omitted_due_to_no_consensus_or_resolver_omission"], 1
        )  # Path is (0,"id")

    def test_dict_revisions_empty_consensus(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.9)  # High threshold
        rev1 = {"key": "val1"}
        rev2 = {"key": "val2"}
        revisions = [rev1, rev2]  # No consensus on path ('key',)
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, {})
        self.assertEqual(analytics["paths_in_consensus_output"], 0)
        self.assertEqual(
            analytics["paths_omitted_due_to_no_consensus_or_resolver_omission"], 1
        )  # Path is ("key",)

    def test_consensus_on_articleinfo_like_structure(self):
        consensus_processor = JSONConsensus(consensus_threshold=0.5)
        rev1 = {
            "_temp_id": "ArticleInfo_common",
            "_type": "ArticleInfo",
            "id": 1,
            "key_topics": ["college football", "Rose Bowl", "racial integration"],
            "raw_text": "The 1956 Rose Bowl was a college football bowl game played on January 2, 1956.",
        }
        rev2 = {  # Differs in key_topics and raw_text
            "_temp_id": "ArticleInfo_common",
            "_type": "ArticleInfo",
            "id": 1,
            "key_topics": ["college football", "Rose Bowl", "Michigan State Spartans"],
            "raw_text": "Michigan State Spartans defeated the UCLA Bruins, 17-14.",
        }
        rev3 = {  # Matches rev1 on differing fields
            "_temp_id": "ArticleInfo_common",
            "_type": "ArticleInfo",
            "id": 1,
            "key_topics": ["college football", "Rose Bowl", "racial integration"],
            "raw_text": "The 1956 Rose Bowl was a college football bowl game played on January 2, 1956.",
        }
        revisions = [rev1, rev2, rev3]
        expected_consensus = {
            "_temp_id": "ArticleInfo_common",
            "_type": "ArticleInfo",
            "id": 1,
            "key_topics": ["college football", "Rose Bowl", "racial integration"],
            "raw_text": "The 1956 Rose Bowl was a college football bowl game played on January 2, 1956.",
        }
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected_consensus)
        # Example: key_topics has 3 elements, so 3 paths. raw_text is 1 path. _temp_id, _type, id are 3 paths. Total 3+1+3 = 7
        self.assertEqual(
            analytics["unique_paths_considered"], 3 + 3 + 1
        )  # _temp_id, _type, id, key_topics.0, key_topics.1, key_topics.2, raw_text
        self.assertEqual(analytics["paths_agreed_by_threshold"], 7)

    def test_consensus_on_wrapped_entities_structure(self):
        consensus_processor = JSONConsensus(
            consensus_threshold=0.51
        )  # Using 0.51 to match orchestrator tests

        # Define content for three revisions of a single entity
        content_rev_a = {
            "_type": "MyModel",
            "_temp_id": "item1",
            "name": "Widget Alpha",
            "value": 100,
            "version": "1.0",
        }
        content_rev_b = {
            "_type": "MyModel",
            "_temp_id": "item1",
            "name": "Widget Alpha",
            "value": 150,
            "version": "1.0",
        }  # value differs
        content_rev_c = {
            "_type": "MyModel",
            "_temp_id": "item1",
            "name": "Widget Alpha",
            "value": 100,
            "version": "1.0",
        }  # value matches A

        # Each revision is wrapped in the {"entities": [...]} structure
        revisions_input = [
            {"entities": [content_rev_a]},
            {"entities": [content_rev_b]},
            {"entities": [content_rev_c]},
        ]

        # Expected consensus content for the entity
        # _type, _temp_id, name, version are unanimous.
        # value: 100 (2/3), 150 (1/3). 2/3 = 0.66 > 0.51, so 100 wins.
        expected_entity_content = {
            "_type": "MyModel",
            "_temp_id": "item1",
            "name": "Widget Alpha",
            "value": 100,
            "version": "1.0",
        }

        # Expected final output from json_consensus
        expected_output_structure = {"entities": [expected_entity_content]}

        consensus_obj, analytics = consensus_processor.get_consensus(revisions_input)

        self.assertEqual(consensus_obj, expected_output_structure)

        # Explicitly check _type and _temp_id in the consensus entity
        self.assertIn("entities", consensus_obj, "Output missing 'entities' key")
        self.assertIsInstance(
            consensus_obj["entities"], list, "'entities' is not a list"
        )
        self.assertEqual(
            len(consensus_obj["entities"]), 1, "Expected one entity in 'entities' list"
        )

        consensus_entity = consensus_obj["entities"][0]
        self.assertIsInstance(
            consensus_entity,
            dict,
            "Extrai entity is not a dictionary",
        )
        self.assertEqual(
            consensus_entity.get("_type"), "MyModel", "_type field mismatch or missing"
        )
        self.assertEqual(
            consensus_entity.get("_temp_id"),
            "item1",
            "_temp_id field mismatch or missing",
        )
        self.assertEqual(
            consensus_entity.get("name"),
            "Widget Alpha",
            "name field mismatch or missing",
        )
        self.assertEqual(
            consensus_entity.get("value"), 100, "value field mismatch or missing"
        )
        self.assertEqual(
            consensus_entity.get("version"), "1.0", "version field mismatch or missing"
        )

        # Analytics checks
        # Paths: (entities,0,_type), (entities,0,_temp_id), (entities,0,name), (entities,0,value), (entities,0,version) = 5 paths
        self.assertEqual(analytics["revisions_processed"], 3)
        self.assertEqual(analytics["unique_paths_considered"], 5)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 5)  # All paths agreed
        self.assertEqual(analytics["paths_resolved_by_conflict_resolver"], 0)
        self.assertEqual(
            analytics["paths_omitted_due_to_no_consensus_or_resolver_omission"], 0
        )
        self.assertEqual(analytics["paths_in_consensus_output"], 5)

    def test_consensus_on_none_value_is_honored(self):
        """
        Tests that None is treated as a valid consensus value when it meets the threshold.
        """
        consensus_processor = JSONConsensus(consensus_threshold=0.6)  # Needs 2/3
        revisions = [{"a": None}, {"a": None}, {"a": 1}]
        # With the fix, 'None' is the most common and its ratio (2/3) > 0.6.
        # So, it should be included in the consensus.
        expected = {"a": None}
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 1)
        self.assertEqual(analytics["paths_in_consensus_output"], 1)

    def test_unanimity_check_logic(self):
        """
        Tests the unanimity check logic when consensus_threshold is 1.0.
        """
        consensus_processor = JSONConsensus(consensus_threshold=1.0)

        # Case 1: Unanimous agreement (is_unanimous=True)
        revisions_unanimous = [{"a": 1}, {"a": 1}]
        expected_unanimous = {"a": 1}
        consensus_obj_u, analytics_u = consensus_processor.get_consensus(
            revisions_unanimous
        )
        self.assertEqual(consensus_obj_u, expected_unanimous)
        self.assertEqual(analytics_u["paths_agreed_by_threshold"], 1)

        # Case 2: Not unanimous agreement (is_unanimous=False)
        revisions_not_unanimous = [{"a": 1}, {"a": 2}]
        expected_not_unanimous = {}
        consensus_obj_nu, analytics_nu = consensus_processor.get_consensus(
            revisions_not_unanimous
        )
        self.assertEqual(consensus_obj_nu, expected_not_unanimous)
        self.assertEqual(analytics_nu["paths_agreed_by_threshold"], 0)
        self.assertEqual(
            analytics_nu["paths_omitted_due_to_no_consensus_or_resolver_omission"], 1
        )

    def test_special_conflict_resolution_for_temp_id_and_type(self):
        # This test ensures that '_temp_id' and '_type' fields have special conflict
        # resolution logic applied, where the most common value is chosen even if
        # the consensus threshold is not met.
        consensus_processor = JSONConsensus(
            consensus_threshold=0.7,  # High threshold to force conflict
            conflict_resolver=default_conflict_resolver,  # Default would omit
        )
        revisions = [
            {"_type": "A", "_temp_id": "id1", "other": "x"},
            {"_type": "A", "_temp_id": "id2", "other": "y"},
            {"_type": "B", "_temp_id": "id1", "other": "z"},
        ]
        # _type: A (2/3), B (1/3). 2/3 = 0.66 < 0.7. Conflict.
        # _temp_id: id1 (2/3), id2 (1/3). 2/3 = 0.66 < 0.7. Conflict.
        # other: x,y,z all 1/3. Conflict.
        # Expected: _type and _temp_id are resolved to most common, 'other' is omitted.
        expected = {"_type": "A", "_temp_id": "id1"}
        consensus_obj, analytics = consensus_processor.get_consensus(revisions)
        self.assertEqual(consensus_obj, expected)
        self.assertEqual(analytics["paths_agreed_by_threshold"], 0)
        # _type and _temp_id are not resolved by the standard conflict_resolver,
        # but by the special internal logic. This is not tracked in analytics
        # under "paths_resolved_by_conflict_resolver".
        self.assertEqual(
            analytics["paths_omitted_due_to_no_consensus_or_resolver_omission"], 1
        )  # 'other'
        self.assertEqual(analytics["paths_in_consensus_output"], 2)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)


class TestConflictResolvers(unittest.TestCase):
    def test_prefer_most_common_resolver_empty_values(self):
        self.assertIsNone(prefer_most_common_resolver(_path=("a",), values=[]))


class TestJSONConsensusInitialization(unittest.TestCase):
    def test_invalid_consensus_threshold(self):
        with self.assertRaisesRegex(
            ValueError,
            "Extrai threshold must be between 0.0 .* and 1.0",
        ):
            JSONConsensus(consensus_threshold=0.0)
        with self.assertRaisesRegex(
            ValueError,
            "Extrai threshold must be between 0.0 .* and 1.0",
        ):
            JSONConsensus(consensus_threshold=-0.1)
        with self.assertRaisesRegex(
            ValueError,
            "Extrai threshold must be between 0.0 .* and 1.0",
        ):
            JSONConsensus(consensus_threshold=1.1)
        # Valid cases (edge)
        try:
            JSONConsensus(consensus_threshold=0.00001)  # very small but > 0
            JSONConsensus(consensus_threshold=1.0)
        except ValueError:
            self.fail(
                "JSONConsensus raised ValueError unexpectedly for valid threshold"
            )
