.. _how_to_generate_sql_model:

How to Generate SQL Models from Text
====================================

This guide provides a step-by-step walkthrough on how to use the ``SQLModelCodeGenerator`` to dynamically create ``SQLModel`` classes from unstructured text.

When to Use This Feature
------------------------

Dynamic model generation is useful when you need a first draft of a data schema based on sample documents. It's a powerful way to bootstrap your data models without writing boilerplate ``SQLModel`` code by hand.

Step 1: Set Up the Generator
----------------------------

First, instantiate the ``SQLModelCodeGenerator`` with your chosen LLM client.

.. code-block:: python

   from extrai.llm_providers import GeminiClient
   from extrai.core import SQLModelCodeGenerator

   # Replace with your actual API key
   llm_client = GeminiClient(api_key="YOUR_GEMINI_API_key")
   generator = SQLModelCodeGenerator(llm_client=llm_client)

Step 2: Define the Context and Task
-----------------------------------

Provide example documents and a clear, natural language description of the models and their relationship.

.. code-block:: python

    import asyncio

    # Input documents that provide context for the schema
    input_docs = [
        "Invoice #INV001 for customer 'ACME Corp' includes 2x 'Super Widget' at $50 each and 1x 'Mega Gadget' at $200.",
        "Invoice #INV002 for 'Beta Inc.', with 3x 'Standard Bolt' at $5 each.",
    ]

    # A natural language description of the required models
    task_description = (
        "Create an Invoice model with an invoice number and customer name. "
        "It should have a relationship to a list of LineItem models. "
        "Each LineItem should have a product name, quantity, and price, and a foreign key to the invoice."
    )

Step 3: Generate and Load the Models
------------------------------------

Call the `generate_and_load_models_via_llm` method to trigger the LLM-based generation. It returns both the loaded classes and the raw Python code.

.. code-block:: python

    async def generate_models():
        loaded_models, generated_code = await generator.generate_and_load_models_via_llm(
            input_documents=input_docs, user_task_description=task_description
        )
        return loaded_models, generated_code

    loaded_models, generated_code = asyncio.run(generate_models())

    print("Generated Code:\n", generated_code)
    print(f"\nSuccessfully loaded models: {list(loaded_models.keys())}")

Step 4: See the Results
-----------------------

The script will output the dynamically generated Python code and a confirmation of the loaded models.

.. code-block:: text

    Generated Code:
    from typing import List, Optional
    from sqlmodel import Field, SQLModel, Relationship


    class LineItem(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        product_name: str
        quantity: int
        price: float
        invoice_id: Optional[int] = Field(default=None, foreign_key="invoice.id")
        invoice: "Invoice" = Relationship(back_populates="line_items")


    class Invoice(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        invoice_number: str
        customer_name: str
        line_items: List["LineItem"] = Relationship(back_populates="invoice")


    Successfully loaded models: ['Invoice', 'LineItem']

You can now save this generated code to a ``.py`` file and import the models directly into your application, providing a powerful head start on your project's data schema.

.. seealso::

   For a complete, runnable script, see the example file: `examples/sqlmodel_generation.py`.
