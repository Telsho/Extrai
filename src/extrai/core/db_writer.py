# random_docs_to_sql/core/db_writer.py
import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Any


class DatabaseWriterError(Exception):
    """Custom exception for database writer errors."""

    pass


def persist_objects(
    db_session: Session, objects_to_persist: List[Any], logger: logging.Logger
) -> None:
    """
    Persists a list of SQLAlchemy objects to the database using the provided session.

    Args:
        db_session: The SQLAlchemy session to use for database operations.
        objects_to_persist: A list of SQLAlchemy model instances to be saved.

    Raises:
        DatabaseWriterError: If an error occurs during the database commit.
    """
    if not objects_to_persist:
        logger.info("No objects provided to persist.")
        return

    try:
        # All objects should already be associated with the session
        # from the hydration phase
        db_session.add_all(objects_to_persist)
        db_session.commit()
        logger.info(
            f"Successfully persisted {len(objects_to_persist)} objects to the database."
        )
    except SQLAlchemyError as e:
        logger.error(f"Database commit failed: {e}", exc_info=True)
        try:
            db_session.rollback()
            logger.info("Database session rolled back successfully.")
        except SQLAlchemyError as rollback_e:
            logger.error(
                f"Failed to rollback database session: {rollback_e}", exc_info=True
            )
            # Potentially raise a more critical error or handle nested failure
        raise DatabaseWriterError(f"Failed to persist objects due to: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during object persistence: {e}",
            exc_info=True,
        )

        if db_session.is_active:
            db_session.rollback()
            logger.info("Database session rolled back due to unexpected error.")

        raise DatabaseWriterError(f"An unexpected error occurred: {e}")
