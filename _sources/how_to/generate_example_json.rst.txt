.. _how_to_generate_example_json:

How to Generate an Example JSON
===============================

This guide explains how to use the ``ExampleJSONGenerator`` to create a valid JSON example from an ``SQLModel``. Providing a good example to the ``WorkflowOr-chestrator`` is one of the most effective ways to improve the accuracy and reliability of data extraction.

When to Use This Feature
------------------------

You should generate an example JSON whenever you don't have one on hand. While the ``WorkflowOrchestrator`` can generate an example automatically, doing so adds an extra LLM call to every extraction task. For production use, it is more cost-effective and consistent to generate the example once and reuse it.

Step 1: Define the Models
-------------------------

First, you need ``SQLModel`` definitions to serve as the schema. This example uses a ``Product`` with related ``ProductSpecs``.

.. code-block:: python

    from typing import Optional
    from sqlmodel import Field, SQLModel, Relationship

    class ProductSpecs(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        display_size: float
        ram_gb: int
        storage_gb: int
        product_id: Optional[int] = Field(default=None, foreign_key="product.id")
        product: Optional["Product"] = Relationship(back_populates="specs")

    class Product(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        price: float
        manufacturer: str
        specs: list["ProductSpecs"] = Relationship(back_populates="product")

Step 2: Initialize the Generator
--------------------------------

Set up the ``ExampleJSONGenerator`` with an LLM client and your root model.

.. code-block:: python

   from extrai.llm_providers import GeminiClient
   from extrai.core import ExampleJSONGenerator

   # Replace with your actual API key
   llm_client = GeminiClient(api_key="YOUR_GEMINI_API_KEY")
   
   generator = ExampleJSONGenerator(llm_client=llm_client, output_model=Product)

Step 3: Generate the Example
----------------------------

Call the ``generate_example`` method to get the JSON string. This method communicates with the LLM and validates the output to ensure it conforms to your model's schema.

.. code-block:: python

   import asyncio
   import json

   async def main():
       example = await generator.generate_example()
       parsed_json = json.loads(example)
       print(json.dumps(parsed_json, indent=2))

   if __name__ == "__main__":
       asyncio.run(main())

Step 4: See the Results
-----------------------

The generated JSON will represent a valid, nested structure that conforms to your ``SQLModel`` definitions.

.. code-block:: json

    [
      {
        "_type": "Product",
        "_temp_id": "prod_001",
        "id": 101,
        "name": "Smartphone X1",
        "price": 799.99,
        "manufacturer": "GlobalTech",
        "specs_ref_ids": [
          "specs_001"
        ]
      },
      {
        "_type": "ProductSpecs",
        "_temp_id": "specs_001",
        "id": 201,
        "display_size": 6.7,
        "ram_gb": 8,
        "storage_gb": 256,
        "product_id": 101,
        "product_ref_id": "prod_001"
      }
    ]

This output can be saved and reused in the ``WorkflowOrchestrator`` to improve extraction accuracy.

.. seealso::

   For a complete, runnable script, see the example file: `examples/example_generation.py`.
