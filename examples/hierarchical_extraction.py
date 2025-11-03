"""
This script demonstrates the hierarchical extraction feature of the library.
It defines a nested data structure (Company -> Department -> Employee) and shows
how the orchestrator can extract entities from text and correctly reconstruct
these complex relationships in the database.
"""

import asyncio
import os
from typing import List, Optional

from extrai.llm_providers import GeminiClient
from sqlalchemy import create_engine
from sqlmodel import Field, Relationship, Session, SQLModel, select

from extrai.core import WorkflowOrchestrator


# ======================================================================================
# 1. Define Hierarchical Data Models
# ======================================================================================
# The key here is the nested relationships: Company has Departments, and
# Departments have Employees. This structure guides the extraction process.


class Employee(SQLModel, table=True):
    """Represents an employee within a department."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    role: str
    department_id: Optional[int] = Field(default=None, foreign_key="department.id")
    department: Optional["Department"] = Relationship(back_populates="employees")


class Department(SQLModel, table=True):
    """Represents a department within a company, containing multiple employees."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    company_id: Optional[int] = Field(default=None, foreign_key="company.id")
    company: Optional["Company"] = Relationship(back_populates="departments")
    employees: List["Employee"] = Relationship(back_populates="department")


class Company(SQLModel, table=True):
    """The root model for the hierarchy."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    industry: str
    departments: List["Department"] = Relationship(back_populates="company")


# ======================================================================================
# 2. Set up LLM Client and Database
# ======================================================================================

# IMPORTANT: Set your Gemini API key as an environment variable
# Example: export GEMINI_API_KEY="your_api_key_here"
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. Please set it to your API key."
    )
llm_client = GeminiClient(api_key=api_key)

# Use an in-memory SQLite database for this example.
engine = create_engine("sqlite:///:memory:")


def reset_database():
    """Clears and recreates the database schema."""
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


# ======================================================================================
# 3. Initialize the Orchestrator for Hierarchical Extraction
# ======================================================================================
# The `use_hierarchical_extraction=True` flag is essential. It tells the orchestrator
# to analyze the relationships between models and extract them level by level.

orchestrator = WorkflowOrchestrator(
    llm_client=llm_client,
    root_sqlmodel_class=Company,
    use_hierarchical_extraction=True,
)

# ======================================================================================
# 4. Define Test Cases
# ======================================================================================
# These text inputs contain nested information that matches our defined hierarchy.

test_cases = [
    {
        "name": "Company with One Department and Two Employees",
        "text": "Innovate Inc., a tech company, has a Research department with two employees: Dr. Alice Smith, the Lead Scientist, and Bob Johnson, a Research Assistant.",
    },
    {
        "name": "Company with Two Departments",
        "text": "Global Corp is in the finance industry. It has a Sales department with Jane Doe as Manager. It also has a Marketing department with John Roe as Director.",
    },
]


# ======================================================================================
# 5. Run Extraction and Print Results
# ======================================================================================


async def main():
    """
    Iterates through test cases, runs the hierarchical extraction,
    and queries the database to show the structured results.
    """
    for i, case in enumerate(test_cases):
        print(f"--- Running Test Case {i + 1}: {case['name']} ---")
        print(f'Input Text: "{case["text"]}"')
        reset_database()
        with Session(engine) as session:
            await orchestrator.synthesize_and_save(
                input_strings=[case["text"]], db_session=session
            )

            # Query for each level of the hierarchy
            companies = session.exec(select(Company)).all()
            departments = session.exec(select(Department)).all()
            employees = session.exec(select(Employee)).all()

            print("\n-- Extracted Data --")
            print("Companies:", companies)
            print("Departments:", departments)
            print("Employees:", employees)
            print("---------------------\n")


if __name__ == "__main__":
    asyncio.run(main())


# ======================================================================================
# 6. Expected Output
# ======================================================================================
"""
Hierarchical extraction is enabled. This may significantly increase LLM API calls and processing time based on model complexity and the number of entities.
--- Running Test Case 1: Company with One Department and Two Employees ---
Extracted Companies: [Company(name='Innovate Inc.', industry='Technology', id=1)]
Extracted Departments: [Department(name='Research', company_id=1, id=1)]
Extracted Employees: [Employee(department_id=1, name='Dr. Alice Smith', id=1, role='Lead Scientist'), Employee(department_id=1, name='Bob Johnson', id=2, role='Research Assistant')]

--- Running Test Case 2: Company with Two Departments ---
Extracted Companies: [Company(name='Global Corp', industry='finance', id=1)]
Extracted Departments: [Department(name='Sales', company_id=1, id=1), Department(name='Marketing', company_id=1, id=2)]
Extracted Employees: [Employee(department_id=1, name='Jane Doe', id=1, role='Manager'), Employee(department_id=2, name='John Roe', id=2, role='Director')]
"""
