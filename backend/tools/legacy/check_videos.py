from sqlmodel import Session, select
from src.db.database import engine, Video

def check_videos():
    with Session(engine) as session:
        videos = session.exec(select(Video)).all()
        print(f"Total videos: {len(videos)}")
        print(f"Pending: {len([v for v in videos if v.status == 'pending'])}")
        print(f"Downloading: {len([v for v in videos if v.status == 'downloading'])}")
        print(f"Transcribing: {len([v for v in videos if v.status == 'transcribing'])}")
        print(f"Processed: {len([v for v in videos if v.status == 'processed'])}") # Assuming status becomes 'processed'
        print(f"Processed (bool): {len([v for v in videos if v.processed])}")
        
        # Print a sample if any are downloading
        downloading = [v for v in videos if v.status == 'downloading']
        if downloading:
            print(f"Sample downloading: {downloading[0].title} ({downloading[0].id})")

if __name__ == "__main__":
    check_videos()
