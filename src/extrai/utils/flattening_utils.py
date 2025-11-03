from typing import Dict, List, Any, Union, Tuple

# Define type aliases for clarity
JSONValue = Union[str, int, float, bool, None]
JSONObject = Dict[str, Any]
JSONArray = List[Any]
Path = Tuple[Union[str, int], ...]
FlattenedJSON = Dict[Path, JSONValue]


def flatten_json(
    nested_json: Union[JSONObject, JSONArray],
    parent_path: Path = (),
    separator: str = ".",
) -> FlattenedJSON:
    """
    Flattens a nested JSON-like dictionary or list into a flat dictionary.

    Keys in the flat dictionary are tuples representing the path to the value.
    List elements are accessed by their integer index in the path.

    Args:
        nested_json: The nested dictionary or list to flatten.
        parent_path: The base path for the current level of recursion.
                     Used internally for recursive calls.
        separator: (Not currently used in tuple-based path, but kept for potential future string path representation)
                   String separator used to join path elements if a string path is desired.

    Returns:
        A flat dictionary where keys are path tuples and values are primitive JSON values.

    Examples:
        >>> flatten_json({'a': 1, 'b': {'c': 2, 'd': [3, 4]}})
        {('a',): 1, ('b', 'c'): 2, ('b', 'd', 0): 3, ('b', 'd', 1): 4}

        >>> flatten_json([{'x': 5}, {'y': 6}])
        {(0, 'x'): 5, (1, 'y'): 6}
    """
    items: FlattenedJSON = {}
    if isinstance(nested_json, dict):
        for k, v in nested_json.items():
            new_path: Path = parent_path + (k,)
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_path, separator))
            else:
                items[new_path] = v
    elif isinstance(nested_json, list):
        for i, v in enumerate(nested_json):
            new_path: Path = parent_path + (i,)
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_path, separator))
            else:
                items[new_path] = v
    else:
        # This case should ideally not be hit if the top-level input is dict or list
        # and other values are primitives. If nested_json is a primitive itself at the top level.
        if parent_path:  # If it's a primitive nested inside, it would have been handled by the caller.
            items[parent_path] = nested_json
        else:  # A single primitive value passed as the top-level json
            items[()] = nested_json

    return items


def unflatten_json(
    flat_json: FlattenedJSON,
) -> Union[JSONObject, JSONArray, JSONValue, None]:
    """
    Unflattens a flat dictionary (with tuple paths) back into a nested JSON-like structure.

    Args:
        flat_json: A flat dictionary where keys are path tuples and values are primitives.

    Returns:
        The reconstructed nested dictionary or list. Returns None for an empty flat_json.
        Returns the single value if the path was empty (e.g. {(): "value"}).

    Examples:
        >>> unflatten_json({('a',): 1, ('b', 'c'): 2, ('b', 'd', 0): 3, ('b', 'd', 1): 4})
        {'a': 1, 'b': {'c': 2, 'd': [3, 4]}}

        >>> unflatten_json({(0, 'x'): 5, (1, 'y'): 6})
        [{'x': 5}, {'y': 6}]

        >>> unflatten_json({(): "hello"})
        "hello"
    """
    if not flat_json:
        return {}  # Or None, depending on desired behavior for empty input

    # Check if the root is a list or dict based on the first key's first element
    # This is a heuristic. A more robust way might involve checking all keys or
    # having an explicit signal if the root was a list.
    # For now, if any path starts with an integer, assume root is a list.
    # If all paths start with strings, assume root is a dict.
    # If there's only one item with an empty path, it's a scalar.

    if len(flat_json) == 1 and () in flat_json:
        return flat_json[()]

    is_root_list = False
    if flat_json:
        first_path_key = next(iter(flat_json.keys()))
        if first_path_key and isinstance(first_path_key[0], int):
            is_root_list = True

    # Determine the maximum index if it's a list to pre-allocate
    max_index = -1
    if is_root_list:
        for path_tuple in flat_json.keys():
            if path_tuple and isinstance(path_tuple[0], int):
                max_index = max(max_index, path_tuple[0])

    # Initialize root based on determined type
    if is_root_list:
        # Initialize list with Nones up to max_index to allow out-of-order path setting
        # This is a simplification. A more robust unflatten would sort keys
        # or build lists dynamically.
        # For list items to be correctly placed, they need to be filled.
        # If paths are like {(0, 'a'): 1, (2, 'b'): 1}, we need list of size 3.
        root: Union[JSONObject, JSONArray] = [None] * (max_index + 1)
    else:
        root = {}

    # Sorting helps in list creation order.
    # Custom key handles paths with mixed types (int for list index, str for dict key)
    # by converting all parts to strings for comparison.
    for path_tuple, value in sorted(
        flat_json.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        current_level = root
        for i, key_part in enumerate(path_tuple):
            is_last_part = i == len(path_tuple) - 1

            if is_last_part:
                if isinstance(current_level, list):
                    # Ensure list is long enough
                    if isinstance(key_part, int) and key_part >= len(current_level):
                        current_level.extend(
                            [None] * (key_part - len(current_level) + 1)
                        )
                    current_level[key_part] = value
                elif isinstance(current_level, dict):
                    current_level[key_part] = value
            else:
                next_key_part = path_tuple[i + 1]
                expected_type = list if isinstance(next_key_part, int) else dict

                if isinstance(current_level, list):
                    if isinstance(key_part, int):
                        if (
                            key_part >= len(current_level)
                            or current_level[key_part] is None
                        ):
                            # Ensure list is long enough
                            if key_part >= len(current_level):
                                current_level.extend(
                                    [None] * (key_part - len(current_level) + 1)
                                )
                            current_level[key_part] = expected_type()
                        elif not isinstance(current_level[key_part], expected_type):
                            raise ValueError(
                                f"Type mismatch at path {path_tuple[: i + 1]}. Expected {expected_type}, found {type(current_level[key_part])}"
                            )
                        current_level = current_level[key_part]
                    else:
                        raise TypeError(
                            f"List index must be int, got {key_part} for path {path_tuple}"
                        )

                elif isinstance(current_level, dict):
                    if key_part not in current_level or current_level[key_part] is None:
                        current_level[key_part] = expected_type()
                    elif not isinstance(current_level[key_part], expected_type):
                        raise ValueError(
                            f"Type mismatch at path {path_tuple[: i + 1]}. Expected {expected_type}, found {type(current_level[key_part])}"
                        )
                    current_level = current_level[key_part]
    return root
