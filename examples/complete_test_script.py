"""
This script demonstrates the core functionality of the extrai library.
It defines a set of SQLModel classes (Product, ProductSpecs, Warranty) to represent product information,
then uses the WorkflowOrchestrator to extract this information from natural language text
and save it into an in-memory SQLite database.
"""

import asyncio
import os
from typing import List, Optional

from extrai.llm_providers import GeminiClient
from sqlalchemy import create_engine
from sqlmodel import Field, Relationship, Session, SQLModel, select

from extrai.core import WorkflowOrchestrator


# ======================================================================================
# 1. Define Your Data Models using SQLModel
# ======================================================================================
# These classes define the structure of the data you want to extract.
# The relationships between them (e.g., a Product having multiple Specs) are crucial.


class ProductSpecs(SQLModel, table=True):
    """Represents a single specification for a product."""

    id: Optional[int] = Field(default=None, primary_key=True)
    spec_name: str
    spec_value: str
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
    """The root model for our extraction. It contains lists of specs and warranties."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    price: float
    manufacturer: str
    specs: List["ProductSpecs"] = Relationship(back_populates="product")
    warranties: List["Warranty"] = Relationship(back_populates="product")


# ======================================================================================
# 2. Set up the LLM Client and Database
# ======================================================================================

# IMPORTANT: Set your Gemini API key as an environment variable
# Example: export GEMINI_API_KEY="your_api_key_here"
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. Please set it to your API key."
    )
llm_client = GeminiClient(api_key=api_key)


# Create an in-memory SQLite database for this example.
# This means the database exists only while the script is running.
engine = create_engine("sqlite:///:memory:")


def reset_database():
    """Clears and recreates the database schema for a clean run."""
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


# ======================================================================================
# 3. Initialize the Workflow Orchestrator
# ======================================================================================
# The orchestrator is the main component that manages the extraction process.
# It takes the LLM client and the root SQLModel class as input.

orchestrator = WorkflowOrchestrator(
    llm_client=llm_client,
    root_sqlmodel_class=Product,
)

# ======================================================================================
# 4. Define Test Cases
# ======================================================================================
# A list of natural language strings from which we want to extract structured data.
# These examples cover simple cases, cases with relationships, and multiple entities.

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
    Iterates through the test cases, runs the extraction process,
    and prints the contents of the database to show the results.
    """
    for i, case in enumerate(test_cases):
        print(f"--- Running Test Case {i + 1}: {case['name']} ---")
        print(f'Input Text: "{case["text"]}"')
        reset_database()
        with Session(engine) as session:
            # This is the core function call that performs the extraction and saves to the DB.
            await orchestrator.synthesize_and_save(
                input_strings=[case["text"]], db_session=session
            )

            # Query the database to see what was extracted
            products = session.exec(select(Product)).all()
            specs = session.exec(select(ProductSpecs)).all()
            warranties = session.exec(select(Warranty)).all()

            print("\n-- Extracted Data --")
            print("Products:", products)
            print("Specs:", specs)
            print("Warranties:", warranties)
            print("---------------------\n")

            # --- Expected Output Placeholder ---
            # --- Running Test Case 1: Simple Product ---
            # Extracted Products: [Product(price=15.5, id=1, name='Basic Mouse', manufacturer='GenericCorp')]
            # Extracted Specs: []
            # Extracted Warranties: []
            # \n
            # --- Running Test Case 2: Product with One Spec ---
            # Extracted Products: [Product(price=120.0, id=1, name='Pro Keyboard', manufacturer='TypeWell')]
            # Extracted Specs: [ProductSpecs(spec_value='Mechanical Switches', product_id=1, spec_name='Mechanical Switches', id=1)]
            # Extracted Warranties: []
            # \n
            # --- Running Test Case 3: Product with Two Specs ---
            # Extracted Products: [Product(price=85.0, id=1, name='Gamer Mouse', manufacturer='SwiftClick')]
            # Extracted Specs: [ProductSpecs(spec_value='16000', product_id=1, spec_name='DPI', id=1), ProductSpecs(spec_value='RGB', product_id=1, spec_name='Lighting', id=2)]
            # Extracted Warranties: []
            # \n
            # --- Running Test Case 4: Product with Specs and Warranties ---
            # Extracted Products: [Product(price=2100.0, id=1, name='Ultra Laptop', manufacturer='PowerPC')]
            # Extracted Specs: [ProductSpecs(spec_value='32GB', product_id=1, spec_name='RAM', id=1), ProductSpecs(spec_value='2TB', product_id=1, spec_name='SSD', id=2)]
            # Extracted Warranties: [Warranty(product_id=1, duration_years=2, support_level='hardware', id=1), Warranty(product_id=1, duration_years=1, support_level='accidental damage', id=2)]
            # \n
            # --- Running Test Case 5: Two Full Products ---
            # Extracted Products: [Product(price=3500.0, id=1, name='DevBook Pro', manufacturer='CodeMasters'), Product(price=950.0, id=2, name='MediaPad 12', manufacturer='StreamCo')]
            # Extracted Specs: [ProductSpecs(spec_value='64GB', product_id=1, spec_name='RAM', id=1), ProductSpecs(spec_value='4TB NVMe SSD', product_id=1, spec_name='Storage', id=2), ProductSpecs(spec_value='12.9-inch', product_id=2, spec_name='Display', id=3), ProductSpecs(spec_value='256GB', product_id=2, spec_name='Storage', id=4)]
            # Extracted Warranties: [Warranty(product_id=1, duration_years=3, support_level='enterprise', id=1), Warranty(product_id=1, duration_years=1, support_level='battery replacement', id=2), Warranty(product_id=2, duration_years=1, support_level='standard', id=3), Warranty(product_id=2, duration_years=1, support_level='screen protection', id=4)]
            # \n


if __name__ == "__main__":
    asyncio.run(main())
