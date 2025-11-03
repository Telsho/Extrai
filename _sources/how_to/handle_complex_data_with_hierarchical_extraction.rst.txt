.. _how_to_handle_complex_data_with_hierarchical_extraction:

How to Handle Complex Data with Hierarchical Extraction
=======================================================

When dealing with complex, nested data structures, a single LLM call can struggle to produce a valid, complete JSON object. The hierarchical extraction feature is designed to solve this by breaking the problem down into smaller, more manageable pieces.

When to Use This Feature
------------------------

Use hierarchical extraction when your data has clear parent-child relationships (e.g., companies, departments, and employees) and standard extraction methods fail to capture the full structure.

.. warning::
   Enabling hierarchical extraction can significantly increase the number of LLM API calls and the total processing time. Expect the number of calls to be roughly `(number of revisions) * (depth of the model)`. For a model with a depth of 3 and 3 revisions, this means about 9 calls, plus one for example generation. It should be used judiciously when the standard extraction method proves insufficient.

Step 1: Define Your Nested Data Models
---------------------------------------

First, define your `SQLModel` schemas, including the relationships between them.

.. code-block:: python

    from typing import Optional, List
    from sqlmodel import Field, SQLModel, Relationship

    class Employee(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        role: str
        department_id: Optional[int] = Field(default=None, foreign_key="department.id")
        department: Optional["Department"] = Relationship(back_populates="employees")

    class Department(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        company_id: Optional[int] = Field(default=None, foreign_key="company.id")
        company: Optional["Company"] = Relationship(back_populates="departments")
        employees: List["Employee"] = Relationship(back_populates="department")

    class Company(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        industry: str
        departments: List["Department"] = Relationship(back_populates="company")

Step 2: Initialize the Orchestrator in Hierarchical Mode
----------------------------------------------------------

To enable this feature, set the `use_hierarchical_extraction` flag to `True` when initializing the `WorkflowOrchestrator`.

.. code-block:: python

    from extrai.core import WorkflowOrchestrator
    # ... other imports

    # Assume llm_client and engine are already configured
    orchestrator = WorkflowOrchestrator(
        llm_client=llm_client,
        root_sqlmodel_class=Company,
        use_hierarchical_extraction=True,
    )

Step 3: Run the Extraction
--------------------------

The call to `synthesize_and_save` remains the same. The orchestrator will handle the step-by-step extraction process automatically.

.. code-block:: python

    import asyncio
    from sqlmodel import Session, select

    async def main():
        text = "Innovate Inc., a tech company, has a Research department with two employees: Dr. Alice Smith, the Lead Scientist, and Bob Johnson, a Research Assistant."
        
        with Session(engine) as session:
            await orchestrator.synthesize_and_save(
                input_strings=[text], db_session=session
            )

            # Query and print the results
            companies = session.exec(select(Company)).all()
            departments = session.exec(select(Department)).all()
            employees = session.exec(select(Employee)).all()

            print("Extracted Companies:", companies)
            print("Extracted Departments:", departments)
            print("Extracted Employees:", employees)

    if __name__ == "__main__":
        asyncio.run(main())

Step 4: See the Results
-----------------------

The output shows that the orchestrator successfully identified and linked the company, its department, and the employees within that department.

.. code-block:: text

    Extracted Companies: [Company(name='Innovate Inc.', industry='tech', id=1)]
    Extracted Departments: [Department(id=1, name='Research', company_id=1)]
    Extracted Employees: [Employee(id=1, name='Dr. Alice Smith', role='Lead Scientist', department_id=1), Employee(id=2, name='Bob Johnson', role='Research Assistant', department_id=1)]

This demonstrates how hierarchical extraction can robustly handle nested data by processing it one level at a time.

.. seealso::

   For a complete, runnable script, see the example file: `examples/hierarchical_extraction.py`.
