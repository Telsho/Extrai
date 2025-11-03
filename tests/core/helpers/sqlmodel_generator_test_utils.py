import ast
from typing import Optional, Dict, Any


# Helper to attempt to compile generated code
def is_valid_python_code(code_string: str) -> bool:
    try:
        compile(code_string, "<string>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:  # Other compilation errors
        return False


def get_node_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):  # Represents variables, True, False, None
        # For True/False/None, node.id would be 'True', 'False', 'None'
        # ast.Constant(True) is preferred in modern Python for these
        return node.id
    elif isinstance(node, ast.Attribute):
        # Reconstructs thing like "uuid.uuid4" or "datetime.utcnow"
        # This assumes the attribute access is not deeper than one level for simplicity here.
        # e.g., module.item, not module.submodule.item
        base = get_node_value(node.value)
        return f"{base}.{node.attr}"
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    ):  # For sa_column_kwargs={}
        # Basic support for dict() calls, assuming empty or simple key-value constants
        if not node.args and not node.keywords:
            return {}
        # This part would need expansion for more complex dicts
    return "UnsupportedAstNodeValue"  # Or raise error for unhandled types


def get_field_call_kwargs(assign_node_value: Optional[ast.expr]) -> Dict[str, Any]:
    if not (
        assign_node_value
        and isinstance(assign_node_value, ast.Call)
        and isinstance(assign_node_value.func, ast.Name)
        and assign_node_value.func.id == "Field"
    ):
        return {}  # Not a Field() call or no value assigned

    kwargs = {}
    for kw in assign_node_value.keywords:
        kwargs[kw.arg] = get_node_value(kw.value)
    return kwargs


def get_annotation_str(ann_node: Optional[ast.AST]) -> str:
    if ann_node is None:
        return ""
    if isinstance(ann_node, ast.Name):
        return ann_node.id
    elif isinstance(
        ann_node, ast.Attribute
    ):  # Handle types like uuid.UUID, datetime.datetime
        # Recursively get the base (e.g., 'uuid' from 'uuid.UUID')
        # and append the attribute part (e.g., '.UUID')
        base_str = get_annotation_str(ann_node.value)
        return f"{base_str}.{ann_node.attr}"
    elif isinstance(ann_node, ast.Subscript):
        value_str = get_annotation_str(ann_node.value)

        # Slice handling for Python 3.8 (ast.Index) vs 3.9+ (direct node or ast.Tuple)
        current_slice = ann_node.slice
        if isinstance(
            current_slice, ast.Index
        ):  # Python 3.8 style for single item slice
            slice_content_node = current_slice.value
        else:  # Python 3.9+ style
            slice_content_node = current_slice

        if isinstance(
            slice_content_node, ast.Tuple
        ):  # For Dict[str, int] or Union[str, int]
            slice_parts = [get_annotation_str(elt) for elt in slice_content_node.elts]
            slice_str = ", ".join(slice_parts)
        else:  # For Optional[int] or List[str]
            slice_str = get_annotation_str(slice_content_node)

        return f"{value_str}[{slice_str}]"
    elif isinstance(ann_node, ast.Constant) and isinstance(
        ann_node.value, str
    ):  # For string literal annotations
        return f'"{ann_node.value}"'  # Should be ann_node.value directly if it's a string literal type
    return "UnsupportedAnnotationType"


def get_class_def_node(tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
    """Finds a class definition node in an AST."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None
