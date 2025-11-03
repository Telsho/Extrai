.. _how_to_using_multiple_llm_providers:

How to Use Multiple LLM Providers
=================================

To improve the robustness and accuracy of data extraction, the ``WorkflowOrchestrator`` supports using multiple LLM providers simultaneously. When multiple clients are provided, the orchestrator will cycle through them, sending requests to each one to generate diverse outputs.

This approach helps mitigate the risk of a single model consistently failing on a specific task and enriches the consensus process.

When to Use This Feature
------------------------

Using multiple LLM providers is beneficial when:

-   You want to maximize the diversity of the generated JSON outputs.
-   You need to improve the reliability of the extraction process.
-   You are not limited by the cost of making additional API calls.

Step 1: Define Your Data Models
-------------------------------

First, define the `SQLModel` schema for the data you want to extract.

.. code-block:: python

    from typing import Optional, List
    from sqlmodel import Field, SQLModel, Relationship

    class ProductSpecs(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        spec_name: str
        spec_value: str
        product_id: Optional[int] = Field(default=None, foreign_key="product.id")
        product: Optional["Product"] = Relationship(back_populates="specs")

    class Product(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        price: float
        manufacturer: str
        specs: List["ProductSpecs"] = Relationship(back_populates="product")

Step 2: Initialize Multiple LLM Clients
---------------------------------------

Instantiate each LLM client you want to use. In this example, we'll set up clients for Gemini and DeepSeek.

.. code-block:: python

    from extrai.llm_providers import GeminiClient, DeepSeekClient

    # Replace with your actual API keys
    gemini_client = GeminiClient(api_key="YOUR_GEMINI_API_KEY")
    deepseek_client = DeepSeekClient(api_key="YOUR_DEEPSEEK_API_KEY")

    llm_clients = [gemini_client, deepseek_client]

Step 3: Initialize the Orchestrator with a List of Clients
----------------------------------------------------------

Instead of passing a single LLM client to the ``WorkflowOrchestrator``, provide the list of clients.

.. code-block:: python

    from extrai.core import WorkflowOrchestrator

    orchestrator = WorkflowOrchestrator(
        llm_client=llm_clients,
        root_sqlmodel_class=Product,
    )

Step 4: Run the Extraction
--------------------------

The rest of the process remains the same. The orchestrator will automatically rotate through the provided clients when making LLM calls.

.. code-block:: python

    import asyncio
    from sqlmodel import Session, select

    async def main():
        text = "Pro Keyboard, $120.00, by TypeWell. Spec: Mechanical Switches."
        
        with Session(engine) as session:
            await orchestrator.synthesize_and_save(
                input_strings=[text], db_session=session
            )

            # Query and print the results
            product = session.exec(select(Product)).all()
            specs = session.exec(select(ProductSpecs)).all()

            print("Extracted Product:", product)
            print("Extracted Specs:", specs)

    if __name__ == "__main__":
        asyncio.run(main())

Step 5: See the Results
-----------------------

The output will be a consolidated result, benefiting from the diverse inputs of multiple LLMs.

.. code-block:: text

    Extracted Product: [Product(price=120.0, id=1, name='Pro Keyboard', manufacturer='TypeWell')]
    Extracted Specs: [ProductSpecs(spec_value='Mechanical Switches', product_id=1, spec_name='Mechanical Switches', id=1)]

By leveraging multiple providers, you can build a more resilient and accurate data extraction pipeline.

.. seealso::

   For a complete, runnable script, see the example file: `examples/rotating_llm_providers.py`.
