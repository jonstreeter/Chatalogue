import sys
from pathlib import Path

from sqlmodel import Session, select

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.src.main as main_mod
from backend.src.db.database import Channel, create_db_and_tables, engine
from backend.src.services.ingestion import IngestionService


def main() -> None:
    create_db_and_tables()
    if main_mod.ingestion_service is None:
        main_mod.ingestion_service = IngestionService()
    with Session(engine) as session:
        rows = session.exec(
            select(Channel.id, Channel.source_type)
            .where((Channel.source_type == None) | (Channel.source_type != "manual"))
            .order_by(Channel.id.asc())
        ).all()
    channel_ids = [int(row[0]) for row in rows if row[0] is not None]
    print(f"Starting metadata backfill for {len(channel_ids)} channels: {channel_ids}", flush=True)
    main_mod._backfill_remote_channel_metadata_task(channel_ids)
    print("Metadata backfill complete.", flush=True)


if __name__ == "__main__":
    main()
