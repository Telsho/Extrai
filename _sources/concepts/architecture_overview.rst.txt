.. _architecture_overview:

Architecture Overview
=====================

The `extrai` library follows a modular, multi-stage pipeline to transform unstructured text into structured, database-ready objects. This document provides an overview of this architecture, covering both the standard and optional dynamic model generation workflows.

Modular Design (Facade Pattern)
-------------------------------

The core `WorkflowOrchestrator` acts as a **Facade**, delegating complex logic to specialized components. This separation of concerns ensures maintainability and extensibility.

*   **ModelRegistry**: Handles schema discovery and model lookup.
*   **ExtractionPipeline**: Orchestrates the core extraction workflow (Standard or Hierarchical).
*   **EntityCounter**: Performs the pre-extraction counting phase to constrain LLM output.
*   **LLMRunner**: Manages LLM client rotation, parallel revision generation, and consensus execution.
*   **JSONConsensus**: The engine for resolving conflicts between LLM revisions.
*   **BatchPipeline**: Manages asynchronous batch job submission, tracking, and processing.
*   **ResultProcessor**: Handles object hydration (converting JSON to SQLModel) and database persistence.

Core Workflow Diagram
---------------------

The following diagram illustrates the complete workflow, including the optional dynamic model generation path.

.. mermaid::

    graph TD
        %% Define styles for different stages for better colors
        classDef inputStyle fill:#f0f9ff,stroke:#0ea5e9,stroke-width:2px,color:#0c4a6e
        classDef processStyle fill:#eef2ff,stroke:#6366f1,stroke-width:2px,color:#3730a3
        classDef consensusStyle fill:#fffbeb,stroke:#f59e0b,stroke-width:2px,color:#78350f
        classDef outputStyle fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d
        classDef modelGenStyle fill:#fdf4ff,stroke:#a855f7,stroke-width:2px,color:#581c87

        subgraph "Inputs (Static Mode)"
            A["📄<br/>Documents"]
            B["🏛️<br/>SQLAlchemy Models"]
            L1["🤖<br/>LLM"]
        end

        subgraph "Inputs (Dynamic Mode)"
            C["📋<br/>Task Description<br/>(User Prompt)"]
            D["📚<br/>Example Documents"]
            L2["🤖<br/>LLM"]
        end

        subgraph "Model Generation<br/>(Optional)"
            MG("🔧<br/>Generate SQLModels<br/>via LLM")
        end

        subgraph "Data Extraction"
            EG("📝<br/>Example Generation<br/>(Optional)")
            CNT("1️⃣<br/>Entity Counting")
            P("✍️<br/>Prompt/Schema Prep")
            
            subgraph "LLM Extraction Revisions"
                direction LR
                E1("🤖<br/>Revision 1")
                E2("🤖<br/>Revision 2")
                E3("🤖<br/>Revision 3")
            end
            
            F("🤝<br/>JSON Consensus")
            H("💧<br/>Hydration (ResultProcessor)")
        end

        subgraph Outputs
            SM["🏛️<br/>Generated SQLModels<br/>(Optional)"]
            O["✅<br/>Hydrated Objects"]
            DB("💾<br/>Database Persistence<br/>(Optional)")
        end

        %% Connections for Static Mode
        L1 --> P
        A --> P
        B --> EG
        EG --> P
        P --> CNT
        CNT --> E1
        CNT --> E2
        CNT --> E3
        E1 --> F
        E2 --> F
        E3 --> F
        F --> H
        H --> O
        H --> DB

        %% Connections for Dynamic Mode
        L2 --> MG
        C --> MG
        D --> MG
        MG --> EG
        EG --> P

        MG --> SM

        %% Apply styles
        class A,B,C,D,L1,L2 inputStyle;
        class P,CNT,E1,E2,E3,H,EG processStyle;
        class F consensusStyle;
        class O,DB,SM outputStyle;
        class MG modelGenStyle;

Component Interaction (Sequence Diagram)
----------------------------------------

This sequence diagram details the interaction between the Facade (`WorkflowOrchestrator`) and its internal subsystems during a standard extraction run.

.. mermaid::

    sequenceDiagram
        participant User
        participant WO as WorkflowOrchestrator
        participant EC as EntityCounter
        participant LLM as LLMRunner
        participant CON as JSONConsensus
        participant RP as ResultProcessor
        participant DB as Database

        User->>WO: synthesize_and_save(text)
        WO->>EC: count_entities(text)
        EC-->>WO: Entity Counts
        
        WO->>LLM: run(text, schema, counts)
        loop Revisions
            LLM->>LLM: Generate Revision 1..N
        end
        LLM-->>WO: List[JSON Revisions]
        
        WO->>CON: consensus(revisions)
        CON-->>WO: Final JSON
        
        WO->>RP: hydrate(json)
        RP-->>WO: SQLModel Objects
        
        WO->>RP: save(objects, session)
        RP->>DB: commit()
        RP-->>WO: Saved Objects
        
        WO-->>User: Saved Objects

Workflow Stages
---------------

The library processes data through the following stages:

0.  **Dynamic Model Generation (Optional)**: In this mode, the `SQLModelCodeGenerator` uses an LLM to generate `SQLModel` class definitions from a high-level task description and example documents. This is ideal when the data schema is not known in advance.

1.  **Documents Ingestion**: The `WorkflowOrchestrator` accepts one or more text documents as the primary input for extraction.

2.  **Schema Introspection**: The `ModelRegistry` inspects the provided `SQLModel` classes (either predefined or dynamically generated) to create a detailed JSON schema or Pydantic model wrapper. This schema is crucial for instructing the LLM on the desired output format (Standard JSON or Structured Output).

3.  **Example Generation (Optional)**: To improve the accuracy of the LLM, the `ExtractionPipeline` can auto-generate few-shot examples from the schema. These examples are included in the prompt to give the LLM a clear template to follow.

4.  **Entity Counting**: Before full extraction, the `EntityCounter` performs a high-level pass to count the expected number of entities. This count is injected into the extraction prompt as a "Critical Quantity Constraint" to improve recall.

5.  **Prompt & Request Preparation**: The `ExtractionRequestFactory` and `PromptBuilder` combine the schema, input documents, entity counts, and examples into a comprehensive prompt. If using "Structured Extraction", this step also prepares the Pydantic models for the LLM's `response_format`.

6.  **LLM Interaction & Revisioning**: The `LLMRunner` rotates through configured clients and sends the prompts to the LLM to produce multiple, independent revisions. This step is fundamental to the consensus mechanism.

7.  **Consensus**: The `JSONConsensus` engine takes all valid revisions and applies a **Weighted Consensus** algorithm. Revisions are weighted by their global similarity to others, and conflicts are resolved field-by-field. Strategies like `SimilarityClusterResolver` ensure that semantic equivalents (e.g., "US" vs "U.S.A.") are correctly unified.

8.  **Object Hydration**: The `ResultProcessor` transforms the final consensus JSON into a graph of `SQLModel` instances. It supports two strategies:
    *   **Direct Hydration**: Recursive instantiation for "Structured Output" results.
    *   **SQLAlchemy Hydration**: Graph reconstruction for flat JSON results (legacy).

9.  **Database Persistence (Optional)**: The hydrated objects are persisted to the database. The `ResultProcessor` performs **Foreign Key Recovery** to ensure relationships are preserved even if Primary Keys were stripped during processing to prevent collisions.
