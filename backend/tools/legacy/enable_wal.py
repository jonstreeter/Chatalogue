from sqlmodel import Session, text
from src.db.database import engine

def enable_wal():
    with Session(engine) as session:
        result = session.exec(text("PRAGMA journal_mode=WAL;")).first()
        print(f"WAL Mode Enable Result: {result}")
        
        # Verify
        current_mode = session.exec(text("PRAGMA journal_mode;")).first()
        print(f"Current Journal Mode: {current_mode}")

if __name__ == "__main__":
    enable_wal()
