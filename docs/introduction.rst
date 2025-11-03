Intro
--------

With `extrai`, you can extract data from text documents with LLMs, which will be formatted into a given `SQLModel` and registered in your database.

The core of the library is its :ref:`Consensus Mechanism <consensus_mechanism>`. We make the same request multiple times, using the same or different providers, and then select the values that meet a certain threshold.

`extrai` also has other features, like :ref:`generating SQLModels <sqlmodel_generator>` from a prompt and documents, and :ref:`generating few-shot examples <example_json_generator>`. For complex, nested data, the library offers :ref:`Hierarchical Extraction <how_to_handle_complex_data_with_hierarchical_extraction>`, breaking down the extraction into manageable, hierarchical steps. It also includes :ref:`built-in analytics <analytics_collector>` to monitor performance and output quality.

Worflow Overview
----------------------

The library is built around a few key components that work together to manage the extraction workflow. The following diagram illustrates the high-level workflow (see :ref:`Architecture Overview <architecture_overview>` for more details):

.. mermaid::

   graph TD
       A[Unstructured Text] --> B(WorkflowOrchestrator);
       C[SQLModel Definition] --> B;
       B --> D{LLM Client};
       D --> E[Multiple JSON Outputs];
       E --> F(SQLAlchemyHydrator);
       F --> G(JSONConsensus);
       G --> H[Consolidated JSON];
       H --> I(SQLAlchemyHydrator);
       I --> J[Structured Data in DB];

Key Features
------------

- :ref:`Consensus Mechanism <consensus_mechanism>`: Improves extraction accuracy by consolidating multiple LLM outputs.
- :ref:`Dynamic SQLModel Generation <sqlmodel_generator>`: Generate `SQLModel` schemas from natural language descriptions.
- :ref:`Hierarchical Extraction <how_to_handle_complex_data_with_hierarchical_extraction>`: Handles complex, nested data by breaking down the extraction into manageable, hierarchical steps.
- :doc:`Extensible LLM Support <api/extrai.llm_providers>`: Integrates with various LLM providers through a client interface.
- :ref:`Built-in Analytics <analytics_collector>`: Collects metrics on LLM performance and output quality to refine prompts and monitor errors.
- :ref:`Workflow Orchestration <workflow_orchestrator>`: A central orchestrator to manage the extraction pipeline.
- :ref:`Example JSON Generation <example_json_generator>`: Automatically generate few-shot examples to improve extraction quality.
- :ref:`Customizable Prompts <how_to_customize_extraction_prompts>`: Customize prompts at runtime to tailor the extraction process to specific needs.
- :ref:`Rotating LLMs providers <how_to_using_multiple_llm_providers>`: Create the JSON revisions from multiple LLM providers.
