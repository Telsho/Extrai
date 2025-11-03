.. _example_json_generator:

Example JSON Generator
======================

The `ExampleJSONGenerator` is a utility component designed to leverage an LLM to generate a valid example JSON string, representing a list of entities, that conforms to a given `SQLModel` schema.

Its primary purpose is to create few-shot examples for the main extraction prompt in the ``WorkflowOrchestrator``. Providing a concrete example helps the LLM understand the desired output format, leading to more accurate data extraction.

Core Workflow
-------------

1.  **Initialization**: You instantiate the ``ExampleJSONGenerator`` with an LLM client and the target `SQLModel` class.
2.  **Execution**: You call the ``generate_example()`` method.
3.  **Processing**: The generator creates a detailed prompt containing the JSON schema of your model and instructs the LLM to produce a single, valid example.
4.  **Validation**: The LLM's output is validated against the `SQLModel` schema to ensure correctness.
5.  **Output**: A valid JSON string representing a list of entities is returned.

Integration with WorkflowOrchestrator
-------------------------------------

The ``WorkflowOrchestrator`` is designed to work with this component.

-   **Automatic Generation**: If you do not provide an ``extraction_example_json`` when calling ``synthesize()`` or ``synthesize_and_save()``, the orchestrator will automatically instantiate and run the ``ExampleJSONGenerator`` to create one.

-   **Recommended Practice**: While automatic generation is convenient, it introduces an extra LLM call every time you run the orchestrator. For consistency and cost-efficiency, we recommend generating an example once, saving it locally (e.g., in a file or as a constant), and passing it explicitly to the orchestrator. This ensures the same example is used for every extraction. See :ref:`how_to_generate_example_json` for a complete example.

Initialization and Configuration
--------------------------------

The constructor configures the generator with the necessary components and validation parameters.

.. code-block:: python

   import logging
   from extrai.core import ExampleJSONGenerator
   from your_models import ProductModel  # Your SQLModel class
   from your_llm_client import llm_client  # An instance of a BaseLLMClient

   # With default logger
   json_generator = ExampleJSONGenerator(
       llm_client=llm_client,
       output_model=ProductModel,
       max_validation_retries_per_revision=1
   )

   # With custom logger
   logger = logging.getLogger("MyCustomLogger")
   json_generator_with_logger = ExampleJSONGenerator(
       llm_client=llm_client,
       output_model=ProductModel,
       logger=logger
   )

**Parameters:**

``llm_client``
   An instance of an LLM client that conforms to the ``BaseLLMClient`` interface.

   *   **Type**: ``BaseLLMClient``

``output_model``
   The `SQLModel` class for which the example JSON will be generated. The generator automatically derives the JSON schema from this model.

   *   **Type**: ``Type[SQLModel]``

``analytics_collector``
   An optional instance for collecting analytics during the generation process.

   *   **Type**: ``Optional[WorkflowAnalyticsCollector]``
   *   **See also**: :doc:`analytics_collector`

``max_validation_retries_per_revision``
   The maximum number of times the generator will ask the LLM to fix its output if it fails schema validation.

   *   **Type**: ``int``
   *   **Default**: ``1``

``logger``
   An optional `logging.Logger` instance. If not provided, a default logger is created.

   *   **Type**: ``Optional[logging.Logger]``
   *   **Default**: ``None``

Core Execution Method
---------------------

The main functionality is exposed through a single async method.

``generate_example()``
   This method orchestrates the entire process of generating and validating the example JSON.

   .. code-block:: python

      try:
          example_json_string = await json_generator.generate_example()
          print("Generated Example:", example_json_string)
      except ExampleGenerationError as e:
          print(f"Failed to generate example: {e}")

   **Returns:**

   *   A valid JSON string (``str``) representing a list of entities that conform to the `output_model` schema.

   **Raises:**

   *   ``ExampleGenerationError``: A wrapper exception that is raised if any part of the process fails, including LLM API errors, validation failures, or unexpected issues. The original exception is attached for debugging.

Practical Example
-----------------

For a complete, step-by-step guide on how to generate, save, and reuse an example, see the following how-to guide:

.. seealso::

   :ref:`how_to_generate_example_json`
      A practical walkthrough of generating and saving a JSON example.

   For a complete, runnable script, see the example file: `examples/example_generation.py`.

For a full API reference, see the :doc:`api/extrai.core` documentation.
