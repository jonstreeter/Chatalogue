from sqlmodel import Session, select
from src.db.database import engine, Job, Video

def fix_zombies():
    with Session(engine) as session:
        # Find running jobs
        zombies = session.exec(select(Job).where(Job.status == "running")).all()
        print(f"Found {len(zombies)} zombie jobs.")
        
        for job in zombies:
            print(f"Resetting Job {job.id} (Video {job.video_id})")
            job.status = "queued"
            job.started_at = None
            session.add(job)
            
            # Reset video status
            video = session.get(Video, job.video_id)
            if video and video.status in ["downloading", "transcribing", "diarizing"]:
                print(f"Resetting Video {video.id} status to pending")
                video.status = "pending"
                session.add(video)
        
        session.commit()
        print("Zombies fixed.")

if __name__ == "__main__":
    fix_zombies()
