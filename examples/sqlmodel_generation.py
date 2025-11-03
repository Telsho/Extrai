"""
This script demonstrates how to use the SQLModelCodeGenerator to dynamically
generate SQLModel Python code from a natural language description and context.
It covers two scenarios: generating a single, simple model and generating
multiple related models.
"""

import asyncio
import os

from extrai.core import SQLModelCodeGenerator
from extrai.llm_providers import GeminiClient

# ======================================================================================
# 1. Set up the LLM Client
# ======================================================================================

# IMPORTANT: Set your Gemini API key as an environment variable
# Example: export GEMINI_API_KEY="your_api_key_here"
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. Please set it to your API key."
    )
llm_client = GeminiClient(api_key=api_key)

# Initialize the generator with the LLM client
generator = SQLModelCodeGenerator(llm_client=llm_client)


# ======================================================================================
# 2. Scenario 1: Generating a Single, Simple Model
# ======================================================================================
print("--- Running Scenario 1: Generating a Single Model ---")

# A simple natural language task description for the model
user_task_simple = "Create a User model with a name, email, and age."
# Contextual documents that help the LLM understand the data
document_context_simple = ["User John Doe is 30 (john.doe@example.com)."]


async def generate_single_model():
    """Generates, loads, and demonstrates a single User model."""
    models, code = await generator.generate_and_load_models_via_llm(
        input_documents=document_context_simple,
        user_task_description=user_task_simple,
    )
    return models, code


# Run the generation process
loaded_models_simple, generated_code_simple = asyncio.run(generate_single_model())
UserModel = loaded_models_simple.get("User")

if UserModel:
    print(f"\nSuccessfully loaded model: {UserModel.__name__}")
    print("Generated code:\n", generated_code_simple)

    # You can now instantiate the dynamically generated model
    user_instance = UserModel(name="Jane Doe", email="jane.doe@example.com", age=28)
    print("Instance:", user_instance)
    print("\n---------------------------------------------------\n")


# ======================================================================================
# 3. Scenario 2: Generating Multiple, Related Models
# ======================================================================================
print("--- Running Scenario 2: Generating Multiple Related Models ---")

# Input documents that provide context for a more complex schema
input_docs_complex = [
    "Invoice #INV001 for customer 'ACME Corp' includes 2x 'Super Widget' at $50 each and 1x 'Mega Gadget' at $200.",
    "Invoice #INV002 for 'Beta Inc.', with 3x 'Standard Bolt' at $5 each.",
]

# A natural language description of the required models and their relationships
task_description_complex = (
    "Create an Invoice model with an invoice number and customer name. "
    "It should have a relationship to a list of LineItem models. "
    "Each LineItem should have a product name, quantity, and price, and a foreign key to the invoice."
)


async def generate_related_models():
    """Generates and loads Invoice and LineItem models."""
    loaded_models, generated_code = await generator.generate_and_load_models_via_llm(
        input_documents=input_docs_complex,
        user_task_description=task_description_complex,
    )
    return loaded_models, generated_code


# Run the generation process
loaded_models_complex, generated_code_complex = asyncio.run(generate_related_models())
print(f"Successfully loaded models: {list(loaded_models_complex.keys())}")

# Access the dynamically created model classes
InvoiceModel = loaded_models_complex.get("Invoice")
LineItemModel = loaded_models_complex.get("LineItem")

if InvoiceModel and LineItemModel:
    print("\nModels generated and loaded successfully!")

    # The `generate_and_load_models_via_llm` method returns the full Python code
    # as a string. We can save this directly to a file.
    output_filename = "generated_models.py"
    with open(output_filename, "w") as f:
        f.write(generated_code_complex)

    print(f"\nSaved generated models to {output_filename}")

    # You can now use the loaded models for data validation, database operations, etc.
    print("\nDemonstrating instantiation of the loaded models:")
    new_invoice = InvoiceModel(
        invoice_number="INV003",
        customer_name="Gamma LLC",
        line_items=[LineItemModel(product_name="Test Item", quantity=1, price=10.0)],
    )
    print(new_invoice)
    print("\n-----------------------------------------------------\n")


# ======================================================================================
# 4. Expected Output
# ======================================================================================
# The output will vary slightly due to the non-deterministic nature of LLMs,
# but it should look similar to the following.

"""
--- Running Scenario 1: Generating a Single Model ---

Successfully loaded model: User
Generated code:
 from sqlmodel import Field, SQLModel
from typing import Optional


class User(SQLModel, table=True):
    \"\"\"Represents a user with a name, email, and age.\"\"\"
    __tablename__ = "users"

    id: Optional[int] = Field(primary_key=True, default=None, nullable=True, description="The unique identifier for the user.")
    name: str = Field(default="", description="The name of the user.")
    email: str = Field(default="", unique=True, description="The email address of the user.")
    age: int = Field(default=0, description="The age of the user.")

Instance: name='Jane Doe' email='jane.doe@example.com' age=28 id=None

---------------------------------------------------

--- Running Scenario 2: Generating Multiple Related Models ---
Successfully loaded models: ['Invoice', 'LineItem']

Models generated and loaded successfully!

Saved generated models to generated_models.py

Demonstrating instantiation of the loaded models:
invoice_number='INV003' customer_name='Gamma LLC' id=None

-----------------------------------------------------

# Content of generated_models.py:
# -----------------------------------------------------
# from sqlmodel import Field, Relationship, SQLModel
# from typing import List, Optional
#
#
# class Invoice(SQLModel, table=True):
#     \"\"\"Represents an invoice with a unique number and customer details.\"\"\"
#     __tablename__ = "invoices"
#
#     id: Optional[int] = Field(primary_key=True, default=None, nullable=True, description="The unique identifier for the invoice.")
#     invoice_number: str = Field(default="", unique=True, index=True, description="Unique identifier for the invoice.")
#     customer_name: str = Field(default="", description="Name of the customer.")
#     line_items: List["LineItem"] = Relationship(back_populates="invoice")
#
#
# class LineItem(SQLModel, table=True):
#     \"\"\"Represents a single item within an invoice.\"\"\"
#     __tablename__ = "line_items"
#
#     id: Optional[int] = Field(primary_key=True, default=None, nullable=True, description="The unique identifier for the line item.")
#     product_name: str = Field(default="", description="Name of the product.")
#     quantity: int = Field(default=0, description="Quantity of the product.")
#     price: float = Field(default=0.0, description="Price per unit of the product.")
#     invoice_id: Optional[int] = Field(default=None, index=True, foreign_key="invoices.id", nullable=True, description="Foreign key to the Invoice model.")
#     invoice: Optional["Invoice"] = Relationship(back_populates="line_items")
# -----------------------------------------------------
"""
