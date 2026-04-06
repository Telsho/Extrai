import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from extrai.core.base_llm_client import ProviderBatchStatus
from extrai.core.batch_models import BatchJobContext, BatchJobStatus
from extrai.core.client_rotator import ClientRotator
from extrai.core.entity_counter import EntityCounter


class BatchStatusChecker:
    def __init__(
        self,
        client_rotator: ClientRotator,
        entity_counter: EntityCounter,
        logger: logging.Logger,
    ):
        self.client_rotator = client_rotator
        self.entity_counter = entity_counter
        self.logger = logger

    async def get_status(
        self, root_batch_id: str, db_session: Session
    ) -> BatchJobStatus:
        context = db_session.get(BatchJobContext, root_batch_id)
        if not context:
            raise ValueError(f"Batch job {root_batch_id} not found")

        terminal_states = [
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
            BatchJobStatus.READY_TO_PROCESS,
            BatchJobStatus.COUNTING_READY_TO_PROCESS,
        ]
        if context.status in terminal_states:
            return context.status

        try:
            # Determine client based on phase
            if context.status in [
                BatchJobStatus.COUNTING_SUBMITTED,
                BatchJobStatus.COUNTING_PROCESSING,
            ]:
                client = self.entity_counter.llm_client
            else:
                client = self.client_rotator.get_next_client()

            provider_status = await client.get_batch_status(
                context.current_batch_id
            )

            if provider_status == ProviderBatchStatus.COMPLETED:
                if context.status in [
                    BatchJobStatus.COUNTING_SUBMITTED,
                    BatchJobStatus.COUNTING_PROCESSING,
                ]:
                    new_status = BatchJobStatus.COUNTING_READY_TO_PROCESS
                else:
                    new_status = BatchJobStatus.READY_TO_PROCESS
            elif provider_status == ProviderBatchStatus.FAILED:
                new_status = BatchJobStatus.FAILED
            elif provider_status == ProviderBatchStatus.CANCELLED:
                new_status = BatchJobStatus.CANCELLED
            elif provider_status == ProviderBatchStatus.PENDING:
                if context.status in [
                    BatchJobStatus.COUNTING_SUBMITTED,
                    BatchJobStatus.COUNTING_PROCESSING,
                ]:
                    new_status = BatchJobStatus.COUNTING_SUBMITTED
                else:
                    new_status = BatchJobStatus.SUBMITTED
            else:  # PROCESSING
                if context.status in [
                    BatchJobStatus.COUNTING_SUBMITTED,
                    BatchJobStatus.COUNTING_PROCESSING,
                ]:
                    new_status = BatchJobStatus.COUNTING_PROCESSING
                else:
                    new_status = BatchJobStatus.PROCESSING

            if new_status != context.status:
                context.status = new_status
                context.updated_at = datetime.now(UTC)
                db_session.add(context)
                db_session.commit()

        except Exception as e:
            self.logger.error(f"Failed to check batch status: {e}", exc_info=True)

        return context.status
