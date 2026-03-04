from sqlmodel import Session, select
from src.db.database import engine, Job

def unsafe_clear_jobs():
    with Session(engine) as session:
        jobs = session.exec(select(Job)).all()
        print(f"Deleting {len(jobs)} jobs...")
        for job in jobs:
            session.delete(job)
        session.commit()
        print("All jobs deleted.")

if __name__ == "__main__":
    unsafe_clear_jobs()
