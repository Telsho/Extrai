# Batch Processing Integration Guide

This document outlines the architecture and usage of the Batch API integration in `WorkflowOrchestrator`. This feature allows for asynchronous, cost-effective extraction workflows using providers like Google Gemini or OpenAI.

## Overview

The Batch Processing integration allows you to offload the heavy lifting of LLM extraction to a background batch process. This is particularly useful for:

*   **Cost Reduction**: Batch APIs often offer significant discounts (e.g., 50% off).
*   **Scalability**: Decoupling submission from processing allows for massive throughput.
*   **Resilience**: The integration includes an automatic state machine that handles retries, hierarchical dependencies, and complex workflows.

## Features

### 1. Structured Output Support
The Batch Pipeline fully supports **Structured Output** (Pydantic models). The system automatically converts your SQLModel classes into the appropriate JSON Schema/Response Format expected by the provider's Batch API.

### 2. Automated Hierarchical Workflows
For complex nested data, the pipeline implements a robust **State Machine** that manages dependencies automatically:

1.  **Counting Phase (`COUNTING_SUBMITTED`)**: (Optional) Submits a batch job to count entities.
2.  **Root Extraction (`SUBMITTED`)**: Extracts the top-level objects using the counts as constraints.
3.  **Child Extraction (`HIERARCHICAL_STEP_SUBMITTED`)**: Recursively submits batch jobs for child entities, linking them to their parents via Foreign Key Recovery.
4.  **Completion (`COMPLETED`)**: Aggregates all results into a final hydrated object graph.

### 3. Smart Resumption and Continuation
If a batch job fails or is interrupted, you don't need to start over.
*   **Monitoring**: Use `monitor_batch_job` to resume tracking an active batch.
*   **Continuation**: Use `create_continuation_batch` to create a new batch that continues from a specific step of a previous one, preserving completed work.

## Usage Guide

### 1. Simple Execution (Managed Loop)
The easiest way to use the batch API is with the `wait_for_completion=True` flag. The orchestrator handles the polling loop for you.

```python
results = await orchestrator.synthesize_batch(
    input_strings=["..."],
    db_session=db_session,
    wait_for_completion=True,
    count_entities=True
)
```

### 2. Manual Integration (e.g., Celery)
For production environments, you may want to manage the polling yourself.

**Phase A: Submission**
```python
job_id = await orchestrator.synthesize_batch(
    input_strings=["..."],
    db_session=db_session,
    count_entities=True,
    wait_for_completion=False
)
# Store job_id in your database
```

**Phase B: Processing Loop**
Periodically check the status and process results. The `process_batch` method is the engine that drives the state machine.

```python
# In your background worker
status = await orchestrator.get_batch_status(job_id, db_session)

if status == BatchJobStatus.READY_TO_PROCESS:
    # Download and process results
    # If this was a hierarchical step, this method automagically submits the NEXT batch!
    result = await orchestrator.process_batch(job_id, db_session)
    
    if result.status == BatchJobStatus.SUBMITTED:
        # A new batch (e.g., for child entities) was started.
        new_provider_id = result.retry_batch_id
        # Update your DB tracking
        
    elif result.status == BatchJobStatus.COMPLETED:
        # The entire workflow is done.
        final_objects = result.hydrated_objects
```

### 3. Resuming a Job
If a job gets stuck or you need to re-run a specific phase (e.g., re-do extraction but keep the counts), use `create_continuation_batch`.

```python
# Restart from hierarchical step 2 (e.g., "Flights" extraction)
new_batch_id = await orchestrator.create_continuation_batch(
    original_batch_id="failed_batch_id",
    db_session=db_session,
    start_from_step_index=2,
    wait_for_completion=True
)
```

## Technical Details

### Context Storage
Batch jobs can involve large contexts (documents + history). The system stores this context in the database using optimized `JSON` column types (mapped to `LONGTEXT` or native `JSON` depending on the DB) to prevent size limit errors.

### Shallow Schema Enforcement
During hierarchical steps, the system uses "Shallow Schemas" (pydantic models without nested relationships) to prevent the LLM from hallucinating deep structures that should be extracted in subsequent steps.
