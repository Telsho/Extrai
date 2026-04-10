# Entity Counting

## Overview

The **Entity Counting** feature performs a high-level pass with the LLM to estimate the number of entities to be extracted *before* the main extraction phase. This count is then injected into the extraction prompt as a "Critical Quantity Constraint", significantly improving recall and reducing hallucinations.

This logic is encapsulated in the `EntityCounter` class, which is automatically managed by the `WorkflowOrchestrator`.

## Key Capabilities

1.  **Recall Improvement**: By explicitly asking "how many items are there?", the LLM is forced to scan the text more thoroughly.
2.  **Hallucination Reduction**: The constraint prevents the LLM from inventing extra entities to fill a quota.
3.  **Context Awareness**: In hierarchical extraction workflows, the counter is aware of previously extracted parent entities, allowing it to provide more accurate counts for child entities (e.g., "3 flights for Traveler A, 2 flights for Traveler B").
4.  **Batch Integration**: Counting is fully integrated into the asynchronous batch pipeline (`COUNTING_SUBMITTED` phase), ensuring scalability.

## Configuration

To enable entity counting, simply pass `count_entities=True` to the synthesis method.

```python
results = await orchestrator.synthesize(
    input_strings=[document_text],
    count_entities=True
)
```

### Customizing the Counting Phase

You can fine-tune the counting process using the following parameters:

**1. Custom Context (`custom_counting_context`)**

Provide specific hints or context for the counting phase. This is useful if the entities are ambiguous.

```python
results = await orchestrator.synthesize(
    ...,
    count_entities=True,
    custom_counting_context="Count 'Invoices' only if they have a non-zero total."
)
```

**2. Sharded / Parallel Counting**

To handle complex counting scenarios (e.g. searching for distinct variations of an entity with disjoint rules), you can pass a `list[str]` of multiple specialized prompts to `custom_counting_context`.

The system automatically executes these prompts concurrently (in "shards"), achieves consensus on each shard individually, and then merges and deduplicates the final results. This parallelization prevents massive monolithic prompts from degrading LLM performance without increasing wall-clock time.

```python
results = await orchestrator.synthesize(
    ...,
    count_entities=True,
    custom_counting_context=[
        "Count only Domestic invoices and describe their destinations.",
        "Count only International invoices and describe their destinations.",
        "Count any catch-all invoices that don't fit the above rules."
    ]
)
```

**3. Dedicated LLM Client (`counting_llm_client`)**

You can use a different model for counting (e.g., a faster/cheaper one) than for the main extraction.

```python
# Initialize orchestrator with a specialized counting client
orchestrator = WorkflowOrchestrator(
    root_sqlmodel_class=MyModel,
    llm_client=gpt4_client,          # Main extraction (high precision)
    counting_llm_client=gpt35_client # Counting (high speed/low cost)
)
```

## How it Works

1.  **Prompting**: The `EntityCounter` generates a specialized prompt asking the LLM to return a JSON object mapping model names to counts (e.g., `{"Invoice": 5, "LineItem": 20}`).
2.  **Execution**: The prompt is sent to the `counting_llm_client` (or the default client).
3.  **Validation**: The output is validated to ensure it matches the expected structure.
4.  **Injection**: The counts are formatted into a constraint string (e.g., *"CRITICAL: You must extract exactly 5 Invoice items..."*) and added to the system prompt of the subsequent extraction phase.
