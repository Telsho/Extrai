.. _architecture_overview:

Architecture Overview
=====================

The `extrai` library follows a modular, multi-stage pipeline to transform unstructured text into structured, database-ready objects. This document provides an overview of this architecture, covering both the standard and optional dynamic model generation workflows.

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
            P("✍️<br/>Prompt Generation")
            
            subgraph "LLM Extraction Revisions"
                direction LR
                E1("🤖<br/>Revision 1")
                H1("💧<br/>SQLAlchemy Hydration 1")
                E2("🤖<br/>Revision 2")
                H2("💧<br/>SQLAlchemy Hydration 2")
                E3("🤖<br/>...")
                H3("💧<br/>...")
            end
            
            F("🤝<br/>JSON Consensus")
            H("💧<br/>SQLAlchemy Hydration")
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
        P --> E1
        P --> E2
        P --> E3
        E1 --> H1
        E2 --> H2
        E3 --> H3
        H1 --> F
        H2 --> F
        H3 --> F
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
        class P,E1,E2,E3,H,EG processStyle;
        class F consensusStyle;
        class O,DB,SM outputStyle;
        class MG modelGenStyle;

Workflow Stages
---------------

The library processes data through the following stages:

0.  **Dynamic Model Generation (Optional)**: In this mode, the `SQLModelCodeGenerator` uses an LLM to generate `SQLModel` class definitions from a high-level task description and example documents. This is ideal when the data schema is not known in advance.

1.  **Documents Ingestion**: The `WorkflowOrchestrator` accepts one or more text documents as the primary input for extraction.

2.  **Schema Introspection**: The library inspects the provided `SQLModel` classes (either predefined or dynamically generated) to create a detailed JSON schema. This schema is crucial for instructing the LLM on the desired output format.

3.  **Example Generation (Optional)**: To improve the accuracy of the LLM, the `ExampleJSONGenerator` can create few-shot examples from the schema. These examples are included in the prompt to give the LLM a clear template to follow.

4.  **Prompt Generation**: The `PromptBuilder` combines the JSON schema, the input documents, and any few-shot examples into a comprehensive system prompt and a user prompt.

5.  **LLM Interaction & Revisioning**: The configured `LLMClient` sends the prompts to the LLM to produce multiple, independent JSON structures (revisions). This step is fundamental to the consensus mechanism.

6.  **JSON Validation & Consensus**: Each JSON revision from the LLM is validated against the schema. The `JSONConsensus` class then takes all valid revisions and applies a consensus algorithm to resolve discrepancies, producing a single, unified JSON object.

7.  **SQLAlchemy Object Hydration**: The `SQLAlchemyHydrator` transforms the final consensus JSON into a graph of `SQLModel` instances, correctly linking related objects.

8.  **Database Persistence (Optional)**: The hydrated `SQLModel` objects can be saved to a relational database via a standard SQLAlchemy session.
