.. _how_to_customize_extraction_prompts:

How to Customize Extraction Prompts
===================================

While the `WorkflowOrchestrator` provides dynamically generated prompts, you can customize them at runtime to improve performance for specific use cases or LLMs.

Instead of changing the entire prompt, the orchestrator allows you to inject custom instructions into specific sections of the prompt by passing arguments to the `synthesize()` or `synthesize_and_save()` methods.

Customizing the Extraction Prompts
----------------------------------

You can modify the prompt's behavior using the following parameters:

-   `custom_extraction_process`: A string that overrides the default step-by-step process the LLM is asked to follow.
-   `custom_extraction_guidelines`: A string that provides a new set of rules or guidelines for the extraction.
-   `custom_final_checklist`: A string that replaces the final checklist the LLM uses to verify its output.
-   `extraction_example_json`: A JSON string that serves as a few-shot example to guide the model's output format.

**Example: Providing Custom Instructions**

Let's say you are extracting data from noisy financial documents and want the LLM to be extra cautious and follow a specific process.

.. code-block:: python

    from extrai.core import WorkflowOrchestrator
    # ... other imports
    import json

    # Assume orchestrator is already initialized
    # orchestrator = WorkflowOrchestrator(...)

    # 1. Define your custom instructions
    process = (
        "1. First, carefully read the entire financial report to understand its context.\n"
        "2. Second, identify the key financial figures based on the schema.\n"
        "3. Third, pay close attention to currency symbols and decimal places.\n"
        "4. Finally, construct the JSON object."
    )

    guidelines = (
        "- Always extract numbers exactly as they appear.\n"
        "- If a value is not present, explicitly use `null`.\n"
        "- Do not infer or calculate any values."
    )

    # 2. Provide a few-shot example for complex cases
    example = {
        "company": "ExampleCorp",
        "revenue": 1500000.50,
        "is_profitable": True
    }
    example_json = json.dumps(example)

    # 3. Call synthesize_and_save with the custom parameters
    unstructured_text = "Financial report text goes here..."
    orchestrator.synthesize_and_save(
        [unstructured_text],
        custom_extraction_process=process,
        custom_extraction_guidelines=guidelines,
        extraction_example_json=example_json
    )

By injecting these targeted instructions, you can significantly influence the LLM's behavior without needing to rewrite the entire prompt structure.
