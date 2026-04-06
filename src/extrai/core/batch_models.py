from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Column
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, Relationship, SQLModel

from enferno.extensions import db

from .config.batch_job_config import BatchJobConfig


class DataClassJSON(TypeDecorator):
    """Custom SQLAlchemy type for dataclasses stored as JSON"""

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: Any | None, dialect) -> dict | None:
        if value is None:
            return None
        if is_dataclass(value):
            return asdict(value)
        return value

    def process_result_value(
        self, value: Any | None, dialect
    ) -> BatchJobConfig | None:
        if value is None:
            return None
        return BatchJobConfig(**value)


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
    metadata = db.metadata

    root_batch_id: str = Field(primary_key=True)
    current_batch_id: str = Field(index=True)  # Provider's batch ID
    status: BatchJobStatus = Field(default=BatchJobStatus.SUBMITTED)

    input_strings: list[str] = Field(default_factory=list, sa_type=JSON)
    config: BatchJobConfig = Field(
        default_factory=BatchJobConfig, sa_column=Column(DataClassJSON)
    )

    # Store results when completed
    results: list[Any] | None = Field(default=None, sa_type=JSON)

    # Tracking retries
    retry_count: int = Field(default=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    # Error tracking
    last_error: str | None = None

    steps: list["BatchJobStep"] = Relationship(back_populates="batch")


class BatchJobStep(SQLModel, table=True):
    metadata = db.metadata

    id: int | None = Field(default=None, primary_key=True)
    batch_id: str = Field(foreign_key="batchjobcontext.root_batch_id")
    step_index: int
    status: BatchJobStatus = Field(default=BatchJobStatus.COMPLETED)
    result: list[Any] = Field(default_factory=list, sa_type=JSON)
    metadata_json: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    batch: BatchJobContext = Relationship(back_populates="steps")


class BatchProcessResult(SQLModel):
    """
    Result returned by process_batch.
    """

    status: BatchJobStatus
    hydrated_objects: list[Any] | None = None
    original_pk_map: dict[Any, Any] | None = Field(default=None, exclude=True)
    retry_batch_id: str | None = None
    message: str | None = None
