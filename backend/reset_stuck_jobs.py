from sqlmodel import Session, select
from src.db.database import engine, Job

def reset_stuck_jobs():
    with Session(engine) as session:
        # Find jobs that are in active states but likely stuck (since we restarted the worker)
        stuck_statuses = ["running", "downloading", "transcribing", "diarizing"]
        jobs = session.exec(select(Job).where(Job.status.in_(stuck_statuses))).all()
        
        count = 0
        for job in jobs:
            print(f"Resetting stuck job {job.id} (Video {job.video_id}) from '{job.status}' to 'queued'")
            job.status = "queued"
            session.add(job)
            count += 1
            
        session.commit()
        print(f"Reset {count} stuck jobs.")

if __name__ == "__main__":
    reset_stuck_jobs()
