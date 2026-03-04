import argparse
from src.db.embedded_postgres import ensure_embedded_postgres, stop_embedded_postgres


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage embedded PostgreSQL runtime.")
    parser.add_argument("action", choices=["start", "stop"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "start":
        ensure_embedded_postgres()
        print("Embedded PostgreSQL is running.")
        return

    stop_embedded_postgres()
    print("Embedded PostgreSQL stopped.")


if __name__ == "__main__":
    main()
