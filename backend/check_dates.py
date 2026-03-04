from sqlmodel import Session, select
from src.db.database import engine, Video

def check_dates():
    with Session(engine) as session:
        videos = session.exec(select(Video).limit(10)).all()
        print(f"Checking {len(videos)} videos:")
        for v in videos:
            print(f"ID: {v.id}, Title: {v.title[:30]}, Published: {v.published_at}")

if __name__ == "__main__":
    check_dates()
