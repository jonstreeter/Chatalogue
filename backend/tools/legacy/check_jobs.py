from sqlmodel import Session, select
from src.db.database import engine, Job, Video

def check_jobs():
    with Session(engine) as session:
        jobs = session.exec(select(Job)).all()
        print(f"Total Jobs Found: {len(jobs)}")
        
        for job in jobs:
            video = session.get(Video, job.video_id)
            title = video.title if video else "Unknown Video"
            print(f"Job ID: {job.id}, Status: {job.status}, Type: {job.job_type}, Video: {title}")

if __name__ == "__main__":
    check_jobs()
