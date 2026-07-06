"""Shared FastAPI dependencies used by main.py and the domain routers."""
from sqlmodel import Session

from .db.database import engine


def get_session():
    with Session(engine) as session:
        yield session


def get_ingestion_service():
    """Late accessor for the ingestion service singleton owned by main.py.

    Imported lazily so router modules never import main.py at module level,
    which would create a circular import.
    """
    from . import main

    return main.ingestion_service
