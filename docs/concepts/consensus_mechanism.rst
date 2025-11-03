.. _consensus_mechanism:

The Consensus Mechanism
=======================

The core of ``extrai`` is its ability to synthesize a single, reliable JSON object from multiple, potentially conflicting JSON outputs (revisions) from an LLM. This process is handled by the ``JSONConsensus`` class.

This page explains how that mechanism works under the hood.

The Core Idea: Field-Level Agreement
------------------------------------

Instead of comparing entire JSON objects, which can be brittle, the consensus mechanism works on a field-by-field basis. It achieves this through a three-step process:

1.  **Flattening**: Each JSON revision is "flattened" into a simple key-value dictionary. Nested structures and list elements are represented using a dot-notation path.
2.  **Aggregation & Voting**: The algorithm aggregates all the values for each unique path across all revisions and determines if any value meets a predefined agreement threshold.
3.  **Un-flattening**: The paths that reached a consensus are used to reconstruct the final, nested JSON object.

Step 1: Flattening
------------------

Consider two JSON revisions for a ``Product`` extraction:

**Revision 1:**

.. code-block:: json

   {
     "name": "SuperWidget",
     "specs": { "ram_gb": 16 },
     "tags": ["A", "B"]
   }

**Revision 2:**

.. code-block:: json

   {
     "name": "SuperWidget",
     "specs": { "ram_gb": 32 },
     "tags": ["A", "C"]
   }

These are flattened into:

- **Revision 1:** ``{"name": "SuperWidget", "specs.ram_gb": 16, "tags.0": "A", "tags.1": "B"}``
- **Revision 2:** ``{"name": "SuperWidget", "specs.ram_gb": 32, "tags.0": "A", "tags.1": "C"}``

Step 2: Aggregation and Voting
------------------------------

The algorithm then groups the values for each path:

- ``name``: ``["SuperWidget", "SuperWidget"]``
- ``specs.ram_gb``: ``[16, 32]``
- ``tags.0``: ``["A", "A"]``
- ``tags.1``: ``["B", "C"]``

Next, it checks each path against the ``consensus_threshold``. This threshold (a float between 0.0 and 1.0) defines the minimum proportion of revisions that must agree. Let's assume a ``consensus_threshold`` of ``0.5``, meaning more than 50% of revisions must agree.

- ``name``: "SuperWidget" appears in 2/2 revisions (100%). **Consensus reached.**
- ``specs.ram_gb``: 16 appears in 1/2 (50%), 32 appears in 1/2 (50%). Neither meets the "> 50%" threshold. **No consensus.**
- ``tags.0``: "A" appears in 2/2 revisions (100%). **Consensus reached.**
- ``tags.1``: "B" appears in 1/2 (50%), "C" appears in 1/2 (50%). **No consensus.**

Step 3: Un-flattening and Conflict Resolution
---------------------------------------------

Only the paths that reached consensus are kept:

- ``name``: "SuperWidget"
- ``tags.0``: "A"

These are then un-flattened to produce the final JSON object:

.. code-block:: json

   {
     "name": "SuperWidget",
     "tags": ["A"]
   }

Notice that ``specs.ram_gb`` and the second tag are missing. This is the default behavior when no consensus is reached for a path.

Conflict Resolution
-------------------

What happens when no value meets the threshold is determined by a ``conflict_resolver`` function that can be passed to the ``JSONConsensus`` initializer. The library provides two main strategies:

-   ``default_conflict_resolver``: If no consensus is found for a path, the field is simply omitted from the final output.
-   ``prefer_most_common_resolver``: If no consensus is found, this resolver will pick the most frequent value, even if it doesn't meet the threshold. This is useful if you always want a value for a field, even if the LLM was inconsistent.

You can also implement your own custom conflict resolver function for more advanced logic.
