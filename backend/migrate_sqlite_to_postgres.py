from pathlib import Path
import argparse

from src.db.database import create_db_and_tables, MIGRATION_MARKER, sqlite_file_name, DATABASE_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate existing SQLite data to PostgreSQL.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete migration marker and re-run migration logic.",
    )
    args = parser.parse_args()

    print(f"Target database: {DATABASE_URL}")
    print(f"Source SQLite: {sqlite_file_name}")

    if not Path(sqlite_file_name).exists():
        print("No SQLite database found. Nothing to migrate.")
        return

    if args.force and MIGRATION_MARKER.exists():
        MIGRATION_MARKER.unlink(missing_ok=True)
        print(f"Deleted migration marker: {MIGRATION_MARKER}")

    create_db_and_tables()
    print("Migration bootstrap complete.")
    if MIGRATION_MARKER.exists():
        print(f"Migration marker: {MIGRATION_MARKER}")


if __name__ == "__main__":
    main()
