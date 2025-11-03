.. _getting_started:

Getting Started: A Step-by-Step Tutorial
=========================================

This tutorial will guide you through the entire process of using `extrai` to extract structured data from a piece of unstructured text. We'll follow a simple, practical example to demonstrate the core workflow.

The Problem: Extracting Product Information
-------------------------------------------

Imagine you have the following product description, and you want to extract key details into a structured format:

*"Gamer Mouse, $85.00, by SwiftClick. Specs: 16000 DPI, RGB Lighting."*

Our goal is to extract the product name, price, manufacturer, and its specifications into a database.

Prerequisites
-------------

Before you begin, make sure you have:

1.  **Python 3.8+** installed.
2.  The `extrai` library installed (see :ref:`installation`).
3.  An **API key** for an LLM provider (e.g., Gemini, OpenAI).

Step 1: Define Your Data Models
-------------------------------

First, we define the structure of the data we want to extract using `SQLModel`. Our models will include `Product`, `ProductSpecs`, and `Warranty`.

.. code-block:: python

    from typing import Optional, List
    from sqlmodel import Field, SQLModel, Relationship

    class ProductSpecs(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        spec_name: str
        spec_value: str
        product_id: Optional[int] = Field(default=None, foreign_key="product.id")
        product: Optional["Product"] = Relationship(back_populates="specs")

    class Warranty(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        duration_years: int
        support_level: str
        product_id: Optional[int] = Field(default=None, foreign_key="product.id")
        product: Optional["Product"] = Relationship(back_populates="warranties")

    class Product(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        price: float
        manufacturer: str
        specs: List["ProductSpecs"] = Relationship(back_populates="product")
        warranties: List["Warranty"] = Relationship(back_populates="product")

Step 2: Set Up the LLM Client and Database
------------------------------------------

Next, we configure the LLM client and set up an in-memory SQLite database for this example.

.. code-block:: python

    from extrai.llm_providers import GeminiClient
    from sqlalchemy import create_engine

    # Replace with your actual API key
    llm_client = GeminiClient(api_key="YOUR_GEMINI_API_KEY")

    # Create an in-memory SQLite database engine
    engine = create_engine("sqlite:///:memory:")

    def reset_database():
        SQLModel.metadata.drop_all(engine)
        SQLModel.metadata.create_all(engine)

Step 3: Initialize the Workflow Orchestrator
--------------------------------------------

The `WorkflowOrchestrator` is the core component that manages the extraction process. We initialize it with the LLM client and our root data model, `Product`.

.. code-block:: python

    from extrai.core import WorkflowOrchestrator

    orchestrator = WorkflowOrchestrator(
        llm_client=llm_client,
        root_sqlmodel_class=Product,
    )

Step 4: Run the Extraction
--------------------------

Now, we can feed unstructured text to the orchestrator. The `synthesize_and_save` method runs the full pipeline: it gets multiple JSON outputs from the LLM, finds a reliable consensus, validates it against your schema, and saves the final objects to the database.

.. code-block:: python

    import asyncio
    from sqlmodel import Session, select

    async def main():
        text = "Gamer Mouse, $85.00, by SwiftClick. Specs: 16000 DPI, RGB Lighting."
        reset_database()
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

When you run the script, the extracted and validated data is saved to the database. The final output will look like this:

.. code-block:: text

    Extracted Product: [Product(name='Gamer Mouse', price=85.0, manufacturer='SwiftClick', id=1)]
    Extracted Specs: [ProductSpecs(id=1, spec_name='DPI', spec_value='16000', product_id=1), ProductSpecs(id=2, spec_name='Lighting', spec_value='RGB', product_id=1)]

You can now explore more advanced topics in our How-to Guides.

.. seealso::

   For a complete, runnable script of this tutorial, you can refer to the example file in the repository: `examples/complete_test_script.py`.
