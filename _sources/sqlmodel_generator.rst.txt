.. _sqlmodel_generator:

SQLModel Generator
==================

The ``SQLModelCodeGenerator`` is a tool that leverages an LLM to automatically generate and load ``SQLModel`` classes directly from your data and a natural language description of your requirements.

.. admonition:: When to Use This Component

   Use this when your data schema is not known in advance or needs to be dynamically generated based on user input or varying document structures. It allows your application to adapt to new data formats without requiring manual code changes.

This component is responsible for:

- Generating a prompt for an LLM based on your documents and task description.
- Interacting with the LLM to get a structured JSON description of the desired data models.
- Generating Python code from this description.
- Dynamically compiling and loading the new ``SQLModel`` classes into your application at runtime.

Core Workflow
-------------

The typical workflow involves these steps:

1.  **Initialization**: You instantiate the ``SQLModelCodeGenerator`` with an LLM client.
2.  **Execution**: You call ``generate_and_load_models_via_llm()`` with your input documents and a natural language description of the models you need.
3.  **Processing**: The generator asks the LLM to design the data models. It then generates Python code for these models.
4.  **Output**: A tuple containing the dynamically loaded ``SQLModel`` classes (as a dictionary) and the generated Python code (as a string).

Let's dive into how to configure and use it.

Initialization and Configuration
--------------------------------

The constructor of the ``SQLModelCodeGenerator`` is straightforward.

.. code-block:: python

   import logging
   from extrai.core import SQLModelCodeGenerator
   from your_llm_client import llm_client  # An instance of a BaseLLMClient

   # Initialize with a default logger
   generator = SQLModelCodeGenerator(llm_client=llm_client)

   # Or with a custom logger
   logger = logging.getLogger("MyCustomLogger")
   generator_with_logger = SQLModelCodeGenerator(
       llm_client=llm_client,
       logger=logger
   )

Here are the parameters you can use:

``llm_client``
   An instance of an LLM client that conforms to the ``BaseLLMClient`` interface. This client will be used to generate the model schema.

   *   **Type**: ``BaseLLMClient``
   *   **Example**:

   .. code-block:: python

      from tests.core.helpers.mock_llm_clients import MockLLMClientSqlGen

      # The client is responsible for the LLM interaction
      generator = SQLModelCodeGenerator(llm_client=MockLLMClientSqlGen())

``analytics_collector``
   An optional instance for collecting detailed analytics. If not provided, a new ``WorkflowAnalyticsCollector`` instance is created automatically.

   *   **Type**: ``Optional[WorkflowAnalyticsCollector]``
   *   **Default**: ``None``
   *   **See also**: For more details, see the :doc:`analytics_collector` documentation.

``logger``
   An optional `logging.Logger` instance. If not provided, a default logger is created.

   *   **Type**: ``Optional[logging.Logger]``
   *   **Default**: ``None``

Core Execution Method
---------------------

Once the generator is configured, you can create models using the ``generate_and_load_models_via_llm()`` method.

``generate_and_load_models_via_llm()``
   This is the primary method that orchestrates the entire dynamic model creation process.

   .. code-block:: python

      loaded_models, generated_code = await generator.generate_and_load_models_via_llm(
          input_documents=["Sample text document..."],
          user_task_description="Create models for users and their posts.",
          num_model_revisions=1,
          max_retries_per_model_revision=2
      )

   **Parameters:**

   *   ``input_documents`` (``List[str]``): A list of text documents that provide context for the data structure. The LLM uses these to infer field names, types, and relationships.
   *   ``user_task_description`` (``str``): A clear, natural language description of the models you want to create. Be as specific as possible.
   *   ``num_model_revisions`` (``int``, optional): The number of different model designs to request from the LLM. The generator will use the first valid one. **Default is ``1``**.
   *   ``max_retries_per_model_revision`` (``int``, optional): The maximum number of retries if the LLM returns an invalid model description that fails schema validation. **Default is ``2``**.

   **Returns:**

   *   A tuple containing:
       *   A dictionary mapping the generated model names (as strings) to the actual, dynamically loaded ``SQLModel`` classes (``Dict[str, Type[SQLModel]]``).
       *   The full, generated source code as a string.

Practical Example
-----------------

For a complete, step-by-step guide on how to generate, use, and persist models, see the following how-to guide:

.. seealso::

   :ref:`how_to_generate_sql_model`
      A practical walkthrough of generating and saving ``SQLModel`` classes.

   For a complete, runnable script, see the example file: `examples/sqlmodel_generation.py`.

For a full API reference, see the :doc:`api/extrai.core` documentation.
