.. _custom_conflict_resolution:

How to Implement Custom Conflict Resolvers
==========================================

When the consensus algorithm encounters conflicting values for a field (e.g., one revision says "Active" and another "Inactive"), and the weighted agreement is below the threshold, it delegates the decision to a **Conflict Resolver**.

By default, `extrai` uses resolvers that either drop the field or pick the most common value. However, you can inject your own logic.

The Resolver Interface
----------------------

A conflict resolver is simply a function with the following signature:

.. code-block:: python

    from typing import Any, List, Optional

    def my_resolver(
        path: str,
        values: List[Any],
        weights: List[float]
    ) -> Optional[Any]:
        ...

*   **path**: The dot-notation path to the field (e.g., `"users.0.status"`).
*   **values**: A list of all candidate values from the different revisions.
*   **weights**: The corresponding trust weights for those values.
*   **Return**: The resolved value, or `None` to omit the field.

Example: Strict Numeric Resolver
--------------------------------

Let's say you are extracting financial data, and if there is *any* disagreement on a price, you want to be conservative and pick the **highest** price found (e.g., for cost estimation), rather than the most common one.

.. code-block:: python

    def conservative_max_price_resolver(path: str, values: List[Any], weights: List[float]) -> Optional[Any]:
        # Only apply custom logic to 'price' fields
        if "price" in path:
            # Filter for numeric values only
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                return max(numeric_values)
        
        # Fallback to default behavior for other fields
        # You can call the default resolver here or return None
        return None

Using Your Resolver
-------------------

Pass your function to the `WorkflowOrchestrator` during initialization.

.. code-block:: python

    from extrai.core import WorkflowOrchestrator

    orchestrator = WorkflowOrchestrator(
        ...,
        conflict_resolver=conservative_max_price_resolver
    )
