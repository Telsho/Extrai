.. _workflow_orchestrator:

Workflow Orchestrator
=====================

The ``WorkflowOrchestrator`` is the central component of the ``extrai`` library. It orchestrates the entire pipeline, from processing raw text documents to generating structured, database-ready SQLModel objects.

This component is responsible for:

- Generating prompts for LLMs based on your data schema.
- Interacting with one or more LLMs to perform data extraction.
- Running a consensus algorithm to merge multiple LLM outputs.
- Hydrating the final JSON data into structured SQLModel objects.
- Persisting these objects to a database session.

Core Workflow
-------------

The typical workflow involves these steps:

1.  **Initialization**: You instantiate the ``WorkflowOrchestrator`` with your root data model (a SQLModel class) and at least one LLM client.
2.  **Execution**: You call either ``synthesize()`` or ``synthesize_and_save()`` with your input documents.
3.  **Processing**: The orchestrator sends the data to the LLM(s), gets back structured JSON, and runs the consensus process.
4.  **Output**: The final, clean data is returned as a list of hydrated SQLModel objects, ready for use in your application or already saved to the database.

Initialization and Configuration
--------------------------------

The constructor of the ``WorkflowOrchestrator`` is key to configuring its behavior.

.. code-block:: python

   from extrai.core import WorkflowOrchestrator

   import logging
   from extrai.core import WorkflowOrchestrator
   from your_models import DepartmentModel  # Your root SQLModel
   from your_llm_client import llm_client  # An instance of a BaseLLMClient

   # Initialize with default logger
   orchestrator = WorkflowOrchestrator(
       root_sqlmodel_class=DepartmentModel,
       llm_client=llm_client,
       num_llm_revisions=3,
       consensus_threshold=0.51,
       # ... other parameters
   )

   # Or with a custom logger
   logger = logging.getLogger("MyCustomLogger")
   orchestrator_with_logger = WorkflowOrchestrator(
       root_sqlmodel_class=DepartmentModel,
       llm_client=llm_client,
       logger=logger
   )

Here are the parameters you can use:

``root_sqlmodel_class``
   The main SQLModel class that serves as the entry point for data extraction. The orchestrator automatically discovers all related SQLModel classes through its relationships.

   *   **Type**: ``Type[SQLModel]``
   *   **Example**:

   .. code-block:: python

      from tests.core.helpers.orchestrator_test_models import DepartmentModel

      # DepartmentModel has a relationship to EmployeeModel,
      # so both will be part of the schema.
      orchestrator = WorkflowOrchestrator(
          root_sqlmodel_class=DepartmentModel,
          llm_client=my_llm_client
      )

``llm_client``
   An instance or a list of instances of an LLM client that conforms to the ``BaseLLMClient`` interface. Providing a list of clients enhances reliability; the orchestrator will rotate through them for each revision.

   *   **Type**: ``Union[BaseLLMClient, List[BaseLLMClient]]``
   *   **Example**:

   .. code-block:: python

      # Single client
      orchestrator = WorkflowOrchestrator(..., llm_client=client1)

      # Multiple clients for resilience
      orchestrator = WorkflowOrchestrator(..., llm_client=[client1, client2])

``num_llm_revisions``
   The total number of times the LLM will be asked to generate a JSON output for the given input. A higher number increases the chances of a reliable consensus but also increases costs and latency.

   *   **Type**: ``int``
   *   **Default**: ``3``
   *   **Example**:

   .. code-block:: python

      # Request 5 different JSON outputs for the consensus process
      orchestrator = WorkflowOrchestrator(..., num_llm_revisions=5)

``max_validation_retries_per_revision``
   The maximum number of times the orchestrator will try to re-validate a single LLM revision if it fails schema validation. This is useful for correcting minor LLM errors automatically.

   *   **Type**: ``int``
   *   **Default**: ``2``

``consensus_threshold``
   The percentage of agreement required for a data point to be included in the final consensus output. For example, a threshold of ``0.51`` means at least 51% of the LLM revisions must agree on a value.

   *   **Type**: ``float``
   *   **Default**: ``0.51``
   *   **Example**:

   .. code-block:: python

      # Require a strict 75% agreement
      orchestrator = WorkflowOrchestrator(..., consensus_threshold=0.75)

``conflict_resolver``
   An optional function to resolve disagreements when the consensus threshold is not met for a specific field. If not provided, a default resolver is used, which typically omits the conflicting field.

   *   **Type**: ``Callable``
   *   **Default**: ``default_conflict_resolver``

``analytics_collector``
   An optional instance for collecting detailed analytics. If not provided, a new ``WorkflowAnalyticsCollector`` instance is created automatically.

   *   **Type**: ``Optional[WorkflowAnalyticsCollector]``
   *   **Default**: ``None``
   *   **See also**: For more details, see the :doc:`analytics_collector` documentation.

``use_hierarchical_extraction``
   If ``True``, enables a more advanced extraction mode designed for deeply nested and complex data models. This mode breaks down the extraction into smaller, manageable parts, which can improve accuracy for complex schemas but may increase the number of LLM calls.

   *   **Type**: ``bool``
   *   **Default**: ``False``
   *   **See also**: For a guide on how to use this feature, see :doc:`how_to/handle_complex_data_with_hierarchical_extraction`.

``logger``
   An optional `logging.Logger` instance. If not provided, a default logger is created.

   *   **Type**: ``Optional[logging.Logger]``
   *   **Default**: ``None``

Core Execution Methods
----------------------

Once the orchestrator is configured, you can start processing documents using one of the two main methods: ``synthesize()`` and ``synthesize_and_save()``.

``synthesize()``
   This method performs the full extraction and consensus pipeline and returns the hydrated SQLModel objects without persisting them to the database. This is useful if you need to perform additional validation or processing before saving.

   .. code-block:: python

      hydrated_objects = await orchestrator.synthesize(
          input_strings=["Text document 1...", "Text document 2..."],
          db_session_for_hydration=db_session  # Optional: for relationship resolution
      )

   **Parameters:**

   *   ``input_strings`` (``List[str]``): A list of strings, where each string is a document to be processed.
   *   ``db_session_for_hydration`` (``Optional[Session]``): An optional SQLAlchemy session. If provided, the hydrator will use it to resolve relationships. If not, a temporary in-memory session is created.
   *   ``extraction_example_json`` (``str``, optional): A JSON string that provides a few-shot example to the LLM, guiding it to produce a better-structured output. If not provided, the orchestrator will attempt to auto-generate one.
   *   ``custom_extraction_process`` (``str``, optional): Custom, step-by-step instructions for the LLM on how to perform the extraction.
   *   ``custom_extraction_guidelines`` (``str``, optional): A list of rules or guidelines for the LLM to follow.
   *   ``custom_final_checklist`` (``str``, optional): A final checklist for the LLM to review before finalizing its output.

``synthesize_and_save()``
   This is the most common method for end-to-end processing. It calls ``synthesize()`` internally and then persists the resulting objects to the database within a single transaction. If any part of the process fails, it automatically rolls back the transaction.

   .. code-block:: python

      # This will extract, hydrate, and save the objects to the DB
      saved_objects = await orchestrator.synthesize_and_save(
          input_strings=["Order confirmation text..."],
          db_session=db_session
      )

   The parameters are the same as for ``synthesize()``, except it requires a ``db_session`` to commit the transaction.

Concise Usage Example
---------------------

This example provides a focused look at initializing and calling the orchestrator, assuming your models and database are already defined. For a full step-by-step guide, please see the :ref:`getting_started` tutorial.

.. code-block:: python

    import asyncio
    from sqlmodel import Session
    from extrai.core import WorkflowOrchestrator

    # Assume the following are already configured:
    # - `YourRootModel`: Your top-level SQLModel class.
    # - `your_llm_client`: An initialized LLM client.
    # - `your_db_engine`: A SQLAlchemy engine.

    # 1. Initialize the WorkflowOrchestrator
    orchestrator = WorkflowOrchestrator(
        root_sqlmodel_class=YourRootModel,
        llm_client=your_llm_client,
        num_llm_revisions=3  # Request 3 revisions for consensus
    )

    # 2. Define the text to process
    unstructured_text = "Some text containing data about a Company and its Employees..."

    # 3. Run the extraction and save the results
    async def run_extraction():
        with Session(your_db_engine) as session:
            saved_objects = await orchestrator.synthesize_and_save(
                [unstructured_text],
                db_session=session
            )
            print(f"Successfully extracted and saved {len(saved_objects)} objects.")

    # Run the asynchronous function
    asyncio.run(run_extraction())
