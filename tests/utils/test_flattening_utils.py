# tests/test_flattening_utils.py
import unittest
from extrai.utils.flattening_utils import (
    flatten_json,
    unflatten_json,
)


class TestFlatteningUtils(unittest.TestCase):
    def test_flatten_empty_dict(self):
        self.assertEqual(flatten_json({}), {})

    def test_unflatten_empty_dict(self):
        self.assertEqual(unflatten_json({}), {})

    def test_flatten_simple_dict(self):
        data = {"a": 1, "b": "hello", "c": True, "d": None}
        expected = {("a",): 1, ("b",): "hello", ("c",): True, ("d",): None}
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_simple_dict(self):
        flat_data = {("a",): 1, ("b",): "hello"}
        expected = {"a": 1, "b": "hello"}
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_nested_dict(self):
        data = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
        expected = {("a",): 1, ("b", "c"): 2, ("b", "d"): 3, ("e",): 4}
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_nested_dict(self):
        flat_data = {("a",): 1, ("b", "c"): 2, ("b", "d"): 3, ("e",): 4}
        expected = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_dict_with_list(self):
        data = {"a": [1, 2], "b": "c"}
        expected = {("a", 0): 1, ("a", 1): 2, ("b",): "c"}
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_dict_with_list(self):
        flat_data = {("a", 0): 1, ("a", 1): 2, ("b",): "c"}
        expected = {"a": [1, 2], "b": "c"}
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_list_of_primitives(self):
        data = [10, "twenty", False]
        expected = {(0,): 10, (1,): "twenty", (2,): False}
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_list_of_primitives(self):
        flat_data = {(0,): 10, (1,): "twenty", (2,): False}
        expected = [10, "twenty", False]
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_list_of_dicts(self):
        data = [{"a": 1}, {"b": 2, "c": [3, 4]}]
        expected = {(0, "a"): 1, (1, "b"): 2, (1, "c", 0): 3, (1, "c", 1): 4}
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_list_of_dicts(self):
        flat_data = {(0, "a"): 1, (1, "b"): 2, (1, "c", 0): 3, (1, "c", 1): 4}
        expected = [{"a": 1}, {"b": 2, "c": [3, 4]}]
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_deeply_nested_structure(self):
        data = {
            "level1_key1": "val1",
            "level1_obj": {
                "level2_key1": "val2",
                "level2_list": [
                    "item1",
                    {"level3_key1": "val3", "level3_key2": "val4"},
                    [{"level4_key1": "val5"}, None],
                ],
            },
        }
        expected = {
            ("level1_key1",): "val1",
            ("level1_obj", "level2_key1"): "val2",
            ("level1_obj", "level2_list", 0): "item1",
            ("level1_obj", "level2_list", 1, "level3_key1"): "val3",
            ("level1_obj", "level2_list", 1, "level3_key2"): "val4",
            ("level1_obj", "level2_list", 2, 0, "level4_key1"): "val5",
            ("level1_obj", "level2_list", 2, 1): None,
        }
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_deeply_nested_structure(self):
        flat_data = {
            ("level1_key1",): "val1",
            ("level1_obj", "level2_key1"): "val2",
            ("level1_obj", "level2_list", 0): "item1",
            ("level1_obj", "level2_list", 1, "level3_key1"): "val3",
            ("level1_obj", "level2_list", 1, "level3_key2"): "val4",
            ("level1_obj", "level2_list", 2, 0, "level4_key1"): "val5",
            ("level1_obj", "level2_list", 2, 1): None,
        }
        expected = {
            "level1_key1": "val1",
            "level1_obj": {
                "level2_key1": "val2",
                "level2_list": [
                    "item1",
                    {"level3_key1": "val3", "level3_key2": "val4"},
                    [{"level4_key1": "val5"}, None],
                ],
            },
        }
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_empty_list(self):
        self.assertEqual(flatten_json([]), {})

    def test_unflatten_to_empty_list_from_empty_flat(self):
        # Current unflatten_json({}) returns {}. To get [], it needs a hint or paths.
        # If the original was [], flatten_json([]) is {}.
        # unflatten_json({}) is {}. This is a known behavior.
        # If we had a path like {(0,): None} then removed it, it might be different.
        # For now, we test the direct unflatten of an empty dict.
        self.assertEqual(unflatten_json({}), {})
        # A more sophisticated unflatten might take an optional original_type hint.

    def test_flatten_list_with_none(self):
        data = ["a", None, "b"]
        expected = {(0,): "a", (1,): None, (2,): "b"}
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_list_with_none(self):
        flat_data = {(0,): "a", (1,): None, (2,): "b"}
        expected = ["a", None, "b"]
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_single_scalar_value(self):
        # Test flattening a single scalar value (not a dict or list)
        self.assertEqual(flatten_json("test_string"), {(): "test_string"})
        self.assertEqual(flatten_json(123), {(): 123})
        self.assertEqual(flatten_json(True), {(): True})
        self.assertEqual(flatten_json(None), {(): None})

    def test_unflatten_single_scalar_value(self):
        self.assertEqual(unflatten_json({(): "test_string"}), "test_string")
        self.assertEqual(unflatten_json({(): 123}), 123)
        self.assertTrue(unflatten_json({(): True}), True)
        self.assertTrue(unflatten_json({(): True}), None)

    def test_unflatten_dict_with_numeric_string_keys(self):
        # Ensure numeric string keys are treated as dict keys, not list indices
        flat_data = {("0",): "val0", ("1", "sub_key"): "val1"}
        expected = {"0": "val0", "1": {"sub_key": "val1"}}
        self.assertEqual(unflatten_json(flat_data), expected)

    def test_flatten_dict_with_numeric_string_keys(self):
        data = {"0": "val0", "1": {"sub_key": "val1"}}
        expected = {("0",): "val0", ("1", "sub_key"): "val1"}
        self.assertEqual(flatten_json(data), expected)

    def test_unflatten_sparse_list(self):
        # Test unflattening where list indices might not be contiguous from 0
        # e.g. if paths like (1,): value exist but (0,): value was removed by consensus
        # The current unflatten_json initializes lists with None up to max_index.
        flat_data = {(1, "a"): "val_a", (3, "b"): "val_b"}
        expected = [None, {"a": "val_a"}, None, {"b": "val_b"}]
        self.assertEqual(unflatten_json(flat_data), expected)

        flat_data_primitives = {(1,): 100, (0,): 50}
        expected_primitives = [50, 100]  # sorted by path before processing
        self.assertEqual(unflatten_json(flat_data_primitives), expected_primitives)

    def test_flatten_primitive_with_parent_path(self):
        # This tests the specific 'if parent_path:' condition in the else block of flatten_json
        # which is hit if flatten_json is called with a primitive and a non-empty parent_path.
        self.assertEqual(flatten_json(123, parent_path=("custom",)), {("custom",): 123})
        self.assertEqual(
            flatten_json("hello", parent_path=("key1", "key2")),
            {("key1", "key2"): "hello"},
        )

    def test_unflatten_error_set_in_primitive_parent(self):
        # Target: line 134 - else: raise TypeError(f"Cannot set value in non-collection type...")
        # Occurs when trying to set a key in a parent that was previously set as a primitive.
        flat_data = {("a",): "scalar_value", ("a", "b"): "child_value"}
        # After ("a",): root = {"a": "scalar_value"}
        # Processing ("a", "b"):
        # path_tuple = ("a", "b"), current_level = root
        # key_part = "a", current_level becomes root["a"] ("scalar_value")
        # key_part = "b", current_level is "scalar_value". is_last_part = True.
        # current_level is not list or dict. TypeError.
        # Actually, this hits ValueError at line 158: Type mismatch at path ('a',). Expected <class 'dict'>, found <class 'str'>
        with self.assertRaisesRegex(
            ValueError,
            "Type mismatch at path \\('a',\\). Expected <class 'dict'>, found <class 'str'>",
        ):
            unflatten_json(flat_data)

    def test_unflatten_error_list_item_type_mismatch(self):
        # Target: line 149 - elif not isinstance(current_level[key_part], expected_type): raise ValueError(...)
        # Occurs when an existing list item is not of the expected collection type for further nesting.
        flat_data = {
            ("a", 0, 0): 1,  # Establishes a[0] as a list: a = [[1]]
            ("a", 0, "k"): 2,  # Tries to treat a[0] as a dict: a = [{k:2}] -> conflict
        }
        # After ("a",0,0): root = {'a': [[1]]}
        # Processing ("a",0,"k"):
        # path_tuple = ("a",0,"k"), current_level = root
        # key_part = "a", current_level = root['a'] ([[[1]]]) -> should be root['a'] which is [[1]]
        # key_part = 0, current_level = root['a'][0] ([1]), next_key_part = "k", expected_type = dict
        # current_level[0] is [1]. isinstance([1], dict) is False. ValueError.
        with self.assertRaisesRegex(
            ValueError,
            "Type mismatch at path \\('a', 0\\). Expected <class 'dict'>, found <class 'list'>",
        ):
            unflatten_json(flat_data)

    def test_unflatten_error_list_index_not_int(
        self,
    ):  # This test actually hits ValueError on line 158
        # Original Target: line 151 (now 154) - else: raise TypeError(f"List index must be int...")
        # Occurs when a non-integer key is used to access/create an element in a list.
        flat_data = {
            ("a", 0): "v1",  # Establishes 'a' as a list: root = {'a': ['v1']}
            ("a", "x", 0): "v2",  # Tries to use string "x" as index for list 'a'
        }
        # Processing ("a","x",0): current_level = root. key_part = "a". next_key_part = "x". expected_type = dict.
        # current_level["a"] is ['v1']. isinstance(['v1'], dict) is False.
        # This raises ValueError at line 158: Type mismatch at path ('a',). Expected <class 'dict'>, found <class 'list'>
        with self.assertRaisesRegex(
            ValueError,
            "Type mismatch at path \\('a',\\). Expected <class 'dict'>, found <class 'list'>",
        ):
            unflatten_json(flat_data)

    def test_unflatten_error_dict_item_type_mismatch(self):
        # Target: line 158 - elif not isinstance(current_level[key_part], expected_type): raise ValueError(...)
        # Occurs when an existing dict item is not of the expected collection type for further nesting.
        flat_data = {
            ("a", "k", 0): 1,  # Establishes a.k as a list: a = {k:[1]}
            (
                "a",
                "k",
                "sk",
            ): 2,  # Tries to treat a.k as a dict: a = {k:{sk:2}} -> conflict
        }
        # After ("a","k",0): root = {'a': {'k': [1]}}
        # Processing ("a","k","sk"):
        # path_tuple = ("a","k","sk"), current_level = root
        # key_part = "a", current_level = root['a'] ({'k':[1]})
        # key_part = "k", current_level = root['a']['k'] ([1]), next_key_part = "sk", expected_type = dict
        # current_level is {'a': {'k':[1]}}. current_level['k'] is [1]. isinstance([1], dict) is False. ValueError.
        with self.assertRaisesRegex(
            ValueError,
            "Type mismatch at path \\('a', 'k'\\). Expected <class 'dict'>, found <class 'list'>",
        ):
            unflatten_json(flat_data)

    def test_unflatten_error_traverse_primitive(self):
        # Target: line 161 - else: raise TypeError(f"Cannot traverse non-collection type...")
        # Occurs when trying to traverse deeper into a path segment that was set as a primitive.
        flat_data = {("a",): "scalar", ("a", "b", "c"): "value"}
        # After ("a",): root = {"a": "scalar"}
        # Processing ("a","b","c"):
        # path_tuple = ("a","b","c"), current_level = root
        # key_part = "a", current_level becomes root["a"] ("scalar")
        # key_part = "b", current_level is "scalar". is_last_part = False.
        # current_level is not list or dict. TypeError.
        # Actually, this hits ValueError at line 158: Type mismatch at path ('a',). Expected <class 'dict'>, found <class 'str'>
        with self.assertRaisesRegex(
            ValueError,
            "Type mismatch at path \\('a',\\). Expected <class 'dict'>, found <class 'str'>",
        ):
            unflatten_json(flat_data)

    def test_unflatten_type_error_list_index_not_int_traversing(self):
        # Target: line 154 - else: raise TypeError(f"List index must be int, got {key_part} for path {path_tuple}")
        # This occurs when current_level is a list, key_part is not an int, and it's not the last part of the path.
        flat_data = {
            (0,): [],  # Establishes root as a list, and root[0] as a list: root = [[]]
            (
                "non_int_idx",
                0,
            ): "value",  # Path starts with non-int, current_level=root (list), key_part="non_int_idx"
        }
        # is_root_list = True because first key (0,) starts with int. root = [[]]
        # Processing ("non_int_idx", 0): current_level = root ([[]])
        # key_part = "non_int_idx". current_level is list. isinstance("non_int_idx", int) is False.
        # Path is ("non_int_idx", 0), so not is_last_part for "non_int_idx".
        # This should hit the TypeError at line 154.
        with self.assertRaisesRegex(
            TypeError,
            "List index must be int, got non_int_idx for path \\('non_int_idx', 0\\)",
        ):
            unflatten_json(flat_data)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
