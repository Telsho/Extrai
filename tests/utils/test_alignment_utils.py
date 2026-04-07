import unittest
import io
import sys
from extrai.utils.alignment_utils import (
    normalize_json_revisions,
    align_entity_arrays,
    find_best_match,
    calculate_similarity,
    compare_values,
)


class TestAlignmentUtils(unittest.TestCase):
    def test_normalize_json_revisions(self):
        """Test normalize_json_revisions with various inputs"""
        cases = [
            {"name": "empty list", "input": [], "expected": []},
            {
                "name": "simple lists reordering",
                "input": [
                    [{"id": 1, "val": "A"}, {"id": 2, "val": "B"}],
                    [{"id": 2, "val": "B"}, {"id": 1, "val": "A"}],
                ],
                "check": lambda res: (
                    len(res) == 2
                    and res[0][0]["id"] == 1
                    and res[1][0]["id"] == 1
                    and res[0][1]["id"] == 2
                    and res[1][1]["id"] == 2
                ),
            },
            {
                "name": "results wrapper",
                "input": [
                    {"results": [{"id": 1, "val": "A"}]},
                    {"results": [{"id": 1, "val": "A"}]},
                ],
                "check": lambda res: (
                    isinstance(res[0], dict)
                    and "results" in res[0]
                    and res[0]["results"][0]["id"] == 1
                ),
            },
            {
                "name": "single objects (no-op)",
                "input": [{"id": 1, "val": "A"}, {"id": 1, "val": "A"}],
                "expected": [{"id": 1, "val": "A"}, {"id": 1, "val": "A"}],
            },
            {
                "name": "mixed empty revisions",
                "input": [[{"id": 1}], []],
                "check": lambda res: len(res) == 2,
            },
            {
                "name": "all empty arrays",
                "input": [[], [], []],
                "expected": [[], [], []],
            },
        ]

        for case in cases:
            with self.subTest(case["name"]):
                result = normalize_json_revisions(case["input"])
                if "expected" in case:
                    self.assertEqual(result, case["expected"])
                if "check" in case:
                    self.assertTrue(case["check"](result))

    def test_align_entity_arrays(self):
        """Test align_entity_arrays with various inputs"""
        cases = [
            {"name": "empty arrays", "input": [], "expected": []},
            {"name": "list of empty arrays", "input": [[], []], "expected": [[], []]},
            {
                "name": "same order",
                "input": [
                    [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                    [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                ],
                "check": lambda res: res[0][0]["id"] == 1 and res[1][0]["id"] == 1,
            },
            {
                "name": "reorder needed",
                "input": [
                    [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                    [{"id": 2, "name": "Bob"}, {"id": 1, "name": "Alice"}],
                ],
                "check": lambda res: (
                    res[0][0]["name"] == "Alice"
                    and res[1][0]["name"] == "Alice"
                    and res[0][1]["name"] == "Bob"
                    and res[1][1]["name"] == "Bob"
                ),
            },
            {
                "name": "no match found (None)",
                "input": [
                    [{"id": 1, "val": "X"}],
                    [{"id": 99, "val": "completely different"}],
                ],
                "check": lambda res: res[0][0]["id"] == 1 and res[1][0] is not None,
            },
            {
                "name": "deeply nested objects",
                "input": [
                    [
                        {"id": 1, "d": {"n": {"v": "deep"}}},
                        {"id": 2, "d": {"n": {"v": "deep"}}},
                    ],
                    [
                        {"id": 2, "d": {"n": {"v": "deep"}}},
                        {"id": 1, "d": {"n": {"v": "deep"}}},
                    ],
                ],
                "check": lambda res: res[0][0]["id"] == 1 and res[1][0]["id"] == 1,
            },
            {
                "name": "lists in objects",
                "input": [
                    [{"id": 1, "tags": ["a", "b"]}, {"id": 2, "tags": ["x", "y"]}],
                    [{"id": 2, "tags": ["x", "y"]}, {"id": 1, "tags": ["a", "b"]}],
                ],
                "check": lambda res: res[0][0]["id"] == 1 and res[1][0]["id"] == 1,
            },
            {
                "name": "three arrays alignment",
                "input": [
                    [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}],
                    [{"name": "Charlie"}, {"name": "Alice"}, {"name": "Bob"}],
                    [{"name": "Bob"}, {"name": "Charlie"}, {"name": "Alice"}],
                ],
                "check": lambda res: (
                    all(r[0]["name"] == "Alice" for r in res)
                    and all(r[1]["name"] == "Bob" for r in res)
                ),
            },
            {
                "name": "preserves reference object identity",
                "input": [[{"id": 1}, {"id": 2}], [{"id": 2}, {"id": 1}]],
                "check": lambda res: (
                    # The first revision (reference) objects should be identical
                    res[0][0]["id"] == 1
                ),
            },
        ]

        for case in cases:
            with self.subTest(case["name"]):
                result = align_entity_arrays(case["input"])
                if "expected" in case:
                    self.assertEqual(result, case["expected"])
                if "check" in case:
                    self.assertTrue(case["check"](result))

    def test_align_different_lengths_warning(self):
        """Test arrays with different lengths (should truncate to min)"""
        arr1 = [{"id": 1}, {"id": 2}, {"id": 3}]
        arr2 = [{"id": 1}, {"id": 2}]

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            aligned = align_entity_arrays([arr1, arr2])
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        self.assertIn("Warning", output)
        self.assertIn("[3, 2]", output)
        self.assertEqual(len(aligned[0]), 2)
        self.assertEqual(len(aligned[1]), 2)

    def test_find_best_match(self):
        """Test find_best_match with various scenarios"""
        cases = [
            {
                "name": "exact id",
                "target": {"id": 5, "name": "Test"},
                "candidates": [
                    {"id": 1, "name": "Other"},
                    {"id": 5, "name": "Test"},
                    {"id": 3, "name": "Another"},
                ],
                "used": set(),
                "expected": 1,
            },
            {
                "name": "skip used indices",
                "target": {"id": 5, "name": "Test"},
                "candidates": [{"id": 5}, {"id": 5}],
                "used": {0},
                "expected": 1,
            },
            {
                "name": "all used",
                "target": {"id": 1},
                "candidates": [{"id": 1}, {"id": 2}],
                "used": {0, 1},
                "expected": -1,
            },
            {
                "name": "by similarity",
                "target": {"name": "Alice", "age": 30, "city": "NYC"},
                "candidates": [
                    {"name": "Bob", "age": 25, "city": "LA"},
                    {"name": "Alice", "age": 30, "city": "Boston"},  # Better match
                    {"name": "Charlie", "age": 40, "city": "Chicago"},
                ],
                "used": set(),
                "expected": 1,
            },
        ]

        for case in cases:
            with self.subTest(case["name"]):
                idx = find_best_match(case["target"], case["candidates"], case["used"])
                self.assertEqual(idx, case["expected"])

    def test_calculate_similarity(self):
        """Test calculate_similarity with various scenarios"""
        cases = [
            ("non-dict strings", "test", "test", 1.0),
            ("non-dict mismatch", "test", "other", 0.0),
            ("both none", None, None, 1.0),
            ("exact id match", {"id": 5, "o": "d"}, {"id": 5, "o": "v"}, 1.0),
            (
                "temp id priority",
                {"_temp_id": "t1", "id": 1},
                {"_temp_id": "t1", "id": 2},
                1.0,
            ),
            ("empty dicts", {}, {}, 1.0),
            ("no common fields", {"a": 1}, {"b": 2}, 0.0),
            (
                "partial match",
                {"a": 1, "b": 2, "c": 3},
                {"a": 1, "b": 2, "c": 999},
                lambda s: 0.5 < s < 1.0,
            ),
            ("missing fields", {"a": 1, "b": 2}, {"a": 1}, 0.5),
            (
                "numeric similarity",
                {"value": 100},
                {"value": 105},
                lambda s: s > calculate_similarity({"value": 100}, {"value": 1000}),
            ),
        ]

        for case in cases:
            name = case[0]
            val1 = case[1]
            val2 = case[2]
            expected = case[3]

            with self.subTest(name):
                score = calculate_similarity(val1, val2)
                if callable(expected):
                    self.assertTrue(expected(score))
                else:
                    self.assertEqual(score, expected)

    def test_compare_values(self):
        """Test compare_values with various types and scenarios"""
        cases = [
            # Basic types
            ("none both", None, None, 1.0),
            ("none one", None, "test", 0.0),
            ("exact int", 42, 42, 1.0),
            ("exact str", "test", "test", 1.0),
            ("exact bool", True, True, 1.0),
            # Strings
            ("case insensitive", "Hello", "hello", 1.0),
            ("string fuzzy", "hello world", "hello word", lambda s: 0.8 < s < 1.0),
            ("string distinct", "abc", "xyz", lambda s: s < 0.3),
            # Numbers
            ("int float equal", 10.0, 10, 1.0),
            ("close numbers", 100, 110, lambda s: 0.8 < s < 1.0),
            ("far numbers", 10, 1000, lambda s: s < 0.5),
            # Booleans
            ("bool mismatch", True, False, 0.0),
            # Lists
            ("empty lists", [], [], 1.0),
            ("one empty list", [], [1], 0.0),
            ("similar lists", [1, 2, 3], [1, 2, 999], lambda s: 0.5 < s < 1.0),
            # Dicts
            ("nested dicts", {"a": 1}, {"a": 1}, 1.0),
            ("nested partial", {"a": 1}, {"a": 2}, lambda s: 0.0 < s < 1.0),
            # Mixed Types (The fix we implemented)
            ("int vs string", 1, "1", 0.0),
            ("list vs dict", [1], {"a": 1}, 0.0),
            ("bool vs int", True, 1, 0.0),
        ]

        for case in cases:
            name = case[0]
            val1 = case[1]
            val2 = case[2]
            expected = case[3]

            with self.subTest(name):
                score = compare_values(val1, val2)
                if callable(expected):
                    self.assertTrue(
                        expected(score), f"Score {score} failed check for {name}"
                    )
                else:
                    self.assertEqual(
                        score, expected, f"Score {score} != {expected} for {name}"
                    )


if __name__ == "__main__":
    unittest.main()
