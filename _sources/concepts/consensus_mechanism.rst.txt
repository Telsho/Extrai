.. _consensus_mechanism:

The Consensus Mechanism
=======================

The core of ``extrai`` is its ability to synthesize a single, reliable JSON object from multiple, potentially conflicting JSON outputs (revisions) from an LLM. This process is handled by the ``JSONConsensus`` class.

This page explains how that mechanism works under the hood.

The Core Idea: Field-Level Agreement
------------------------------------

Instead of comparing entire JSON objects, which can be brittle, the consensus mechanism works on a field-by-field basis. It achieves this through a three-step process:

1.  **Flattening**: Each JSON revision is "flattened" into a simple key-value dictionary. Nested structures and list elements are represented using a dot-notation path.
2.  **Weighted Aggregation**: The algorithm calculates a "Trust Score" for each revision based on its similarity to others, then aggregates values for each path.
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

Step 2: Weighted Aggregation and Voting
---------------------------------------

Unlike simple majority voting, `extrai` uses a **Weighted Consensus** algorithm.

1.  **Trust Score Calculation**: Each revision is compared against all others (using Levenshtein similarity). Revisions that are more similar to the group average get a higher weight. This helps filter out "hallucinations" or "lazy" responses that diverge significantly from the consensus.
2.  **Vote Aggregation**: The algorithm groups values for each path and sums their weights.

For example, if Revision 1 is deemed "more trustworthy" (weight 1.2) and Revision 2 is "less trustworthy" (weight 0.8):

- ``name`` ("SuperWidget"): 1.2 + 0.8 = 2.0 (Total Agreement)
- ``tags.1``:
    - "B": 1.2
    - "C": 0.8

The system then checks if the **Weighted Agreement Ratio** (value weight / total weight) meets the ``consensus_threshold``.

Step 3: Conflict Resolution & Clustering
----------------------------------------

What happens when no value meets the threshold? The system employs a `conflict_resolver`.

**Standard Resolvers**:

-   ``default_conflict_resolver``: Omits the field if no consensus is reached.
-   ``prefer_most_common_resolver``: Picks the value with the highest weight, even if it's below the threshold.

**Advanced: Similarity Cluster Resolver**

String fields often suffer from minor formatting differences that cause false conflicts. The ``SimilarityClusterResolver`` handles this by clustering similar values.

**Example Scenario**:
Three LLM revisions extract a "Country" field:

1.  "USA"
2.  "U.S.A."
3.  "United States"
4.  "France"

Without clustering, each value might have a low agreement score (e.g., 25% each), failing the threshold.

With **Similarity Clustering**:

1.  The resolver detects that "USA" and "U.S.A." are highly similar (Levenstein distance).
2.  "United States" might also be linked via semantic matching (if enabled).
3.  "France" is distinct.
4.  The system treats {"USA", "U.S.A."} as a single consensus group with 50% agreement (assuming equal weights).
5.  If this meets the threshold, the most standard format (e.g., "USA") is selected as the final value.

Analytics & Metrics
-------------------

The process produces detailed metrics available in the `WorkflowAnalyticsCollector`:

-   **`consensus_confidence_score`**: An aggregate score (0.0 - 1.0) indicating how "confident" the system is in the final result, based on the average agreement ratio across all fields.
-   **`average_string_similarity`**: Measures how textually similar the revisions were to each other.
