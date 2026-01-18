from enum import Enum
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import JSON


class BatchJobStatus(str, Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    READY_TO_PROCESS = "ready_to_process"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # Counting phase statuses
    COUNTING_SUBMITTED = "counting_submitted"
    COUNTING_PROCESSING = "counting_processing"
    COUNTING_READY_TO_PROCESS = "counting_ready_to_process"


class BatchJobContext(SQLModel, table=True):
    """
    Stores the state of a batch job managed by the WorkflowOrchestrator.
    """

    root_batch_id: str = Field(primary_key=True)
    current_batch_id: str = Field(index=True)  # Provider's batch ID
    status: BatchJobStatus = Field(default=BatchJobStatus.SUBMITTED)

    input_strings: List[str] = Field(default_factory=list, sa_type=JSON)
    config: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)

    # Store results when completed
    results: Optional[List[Any]] = Field(default=None, sa_type=JSON)

    # Tracking retries
    retry_count: int = Field(default=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Error tracking
    last_error: Optional[str] = None

    steps: List["BatchJobStep"] = Relationship(back_populates="batch")


class BatchJobStep(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: str = Field(foreign_key="batchjobcontext.root_batch_id")
    step_index: int
    status: BatchJobStatus = Field(default=BatchJobStatus.COMPLETED)
    result: List[Any] = Field(default_factory=list, sa_type=JSON)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    batch: BatchJobContext = Relationship(back_populates="steps")


class BatchProcessResult(SQLModel):
    """
    Result returned by process_batch.
    """

    status: BatchJobStatus
    hydrated_objects: Optional[List[Any]] = None
    original_pk_map: Optional[Dict[Any, Any]] = Field(default=None, exclude=True)
    retry_batch_id: Optional[str] = None
    message: Optional[str] = None
