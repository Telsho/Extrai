"""
This script demonstrates how to use multiple LLM providers in a rotating fashion
for the data extraction workflow. By passing a list of LLM clients to the
WorkflowOrchestrator, the system will cycle through them for each extraction task.
This can improve robustness and reduce reliance on a single provider.
"""

import asyncio
import os
from typing import List, Optional

from extrai.llm_providers import DeepSeekClient, GeminiClient
from sqlalchemy import create_engine
from sqlmodel import Field, Relationship, Session, SQLModel, select

from extrai.core import WorkflowOrchestrator


# ======================================================================================
# 1. Define Your Data Models using SQLModel
# ======================================================================================
# These classes define the structure of the data you want to extract.


class ProductSpecs(SQLModel, table=True):
    """Represents a single specification for a product."""

    id: Optional[int] = Field(default=None, primary_key=True)
    spec_name: str
    spec_value: str = Field(default="", description="Spec value, for example 30GB")
    product_id: Optional[int] = Field(default=None, foreign_key="product.id")
    product: Optional["Product"] = Relationship(back_populates="specs")


class Warranty(SQLModel, table=True):
    """Represents a warranty for a product."""

    id: Optional[int] = Field(default=None, primary_key=True)
    duration_years: int
    support_level: str
    product_id: Optional[int] = Field(default=None, foreign_key="product.id")
    product: Optional["Product"] = Relationship(back_populates="warranties")


class Product(SQLModel, table=True):
    """The root model for our extraction."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    price: float
    manufacturer: str
    specs: List["ProductSpecs"] = Relationship(back_populates="product")
    warranties: List["Warranty"] = Relationship(back_populates="product")


# ======================================================================================
# 2. Set up the LLM Clients and Database
# ======================================================================================

# IMPORTANT: Set your API keys as environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not gemini_api_key or not deepseek_api_key:
    raise ValueError(
        "Both GEMINI_API_KEY and DEEPSEEK_API_KEY environment variables must be set."
    )

# Initialize clients with their respective API keys
gemini_client = GeminiClient(api_key=gemini_api_key)
deepseek_client = DeepSeekClient(api_key=deepseek_api_key)

# Create a list of clients to be used in rotation
llm_clients = [gemini_client, deepseek_client]

# Create an in-memory SQLite database for this example
engine = create_engine("sqlite:///:memory:")


def reset_database():
    """Clears and recreates the database schema for a clean run."""
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


# ======================================================================================
# 3. Initialize the Workflow Orchestrator
# ======================================================================================
# The orchestrator is configured with the list of LLM clients.

orchestrator = WorkflowOrchestrator(llm_client=llm_clients, root_sqlmodel_class=Product)

# ======================================================================================
# 4. Define Test Cases
# ======================================================================================

test_cases = [
    {"name": "Simple Product", "text": "Basic Mouse, $15.50, by GenericCorp."},
    {
        "name": "Product with One Spec",
        "text": "Pro Keyboard, $120.00, by TypeWell. Spec: Mechanical Switches.",
    },
    {
        "name": "Product with Two Specs",
        "text": "Gamer Mouse, $85.00, by SwiftClick. Specs: 16000 DPI, RGB Lighting.",
    },
    {
        "name": "Product with Specs and Warranties",
        "text": "Ultra Laptop, $2100.00, by PowerPC. Specs: 32GB RAM, 2TB SSD. Warranties: 2-year hardware, 1-year accidental damage.",
    },
    {
        "name": "Two Full Products",
        "text": "Product A: DevBook Pro, $3500.00, by CodeMasters. Specs: 64GB RAM, 4TB NVMe SSD. Warranties: 3-year enterprise support, 1-year battery replacement. Product B: MediaPad 12, $950.00, by StreamCo. Specs: 12.9-inch display, 256GB Storage. Warranties: 1-year standard, 1-year screen protection.",
    },
]


# ======================================================================================
# 5. Run the Extraction and Print Results
# ======================================================================================


async def main():
    """
    Iterates through test cases, runs the extraction, and prints the results.
    """
    for i, case in enumerate(test_cases):
        print(f"--- Running Test Case {i + 1}: {case['name']} ---")
        reset_database()
        with Session(engine) as session:
            # The orchestrator will use the LLM clients in rotation.
            await orchestrator.synthesize_and_save(
                input_strings=[case["text"]], db_session=session
            )

            # Query and print the extracted data
            products = session.exec(select(Product)).all()
            specs = session.exec(select(ProductSpecs)).all()
            warranties = session.exec(select(Warranty)).all()

            print("Extracted Products:", products)
            print("Extracted Specs:", specs)
            print("Extracted Warranties:", warranties)
        print("\\n")


if __name__ == "__main__":
    asyncio.run(main())

# ======================================================================================
# 6. Expected Output
# ======================================================================================
"""
--- Running Test Case 1: Simple Product ---
Extracted Products: [Product(name='Basic Mouse', manufacturer='GenericCorp', price=15.5, id=1)]
Extracted Specs: []
Extracted Warranties: []

--- Running Test Case 2: Product with One Spec ---
Extracted Products: [Product(name='Pro Keyboard', manufacturer='TypeWell', price=120.0, id=1)]
Extracted Specs: [ProductSpecs(spec_name='Mechanical Switches', product_id=1, id=1, spec_value='Mechanical Switches')]
Extracted Warranties: []

--- Running Test Case 3: Product with Two Specs ---
Extracted Products: [Product(name='Gamer Mouse', manufacturer='SwiftClick', price=85.0, id=1)]
Extracted Specs: [ProductSpecs(spec_name='DPI', product_id=1, id=1, spec_value='16000'), ProductSpecs(spec_name='Lighting', product_id=1, id=2, spec_value='RGB')]
Extracted Warranties: []

--- Running Test Case 4: Product with Specs and Warranties ---
Extracted Products: [Product(name='Ultra Laptop', manufacturer='PowerPC', price=2100.0, id=1)]
Extracted Specs: [ProductSpecs(spec_name='RAM', product_id=1, id=1, spec_value='32GB'), ProductSpecs(spec_name='SSD', product_id=1, id=2, spec_value='2TB')]
Extracted Warranties: [Warranty(support_level='hardware', id=1, product_id=1, duration_years=2), Warranty(support_level='accidental damage', id=2, product_id=1, duration_years=1)]

--- Running Test Case 5: Two Full Products ---
Extracted Products: [Product(name='DevBook Pro', manufacturer='CodeMasters', price=3500.0, id=1), Product(name='MediaPad 12', manufacturer='StreamCo', price=950.0, id=2)]
Extracted Specs: [ProductSpecs(spec_name='RAM', product_id=1, id=1, spec_value='64GB'), ProductSpecs(spec_name='Storage', product_id=1, id=2, spec_value='4TB NVMe SSD'), ProductSpecs(spec_name='Display', product_id=2, id=3, spec_value='12.9-inch'), ProductSpecs(spec_name='Storage', product_id=2, id=4, spec_value='256GB')]
Extracted Warranties: [Warranty(support_level='enterprise', id=1, product_id=1, duration_years=3), Warranty(support_level='battery replacement', id=2, product_id=1, duration_years=1), Warranty(support_level='standard', id=3, product_id=2, duration_years=1), Warranty(support_level='screen protection', id=4, product_id=2, duration_years=1)]
"""
