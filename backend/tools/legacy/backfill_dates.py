from sqlmodel import Session, select
from src.db.database import engine, Video
import yt_dlp
from datetime import datetime

def backfill_dates():
    with Session(engine) as session:
        videos = session.exec(select(Video).where(Video.published_at == None)).all()
        print(f"Found {len(videos)} videos with missing dates.")
        
        ydl_opts = {
            'quiet': True,
            'ignoreerrors': True,
        }
        
        count = 0
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for video in videos:
                print(f"Backfilling {video.title} ({video.youtube_id})...")
                url = f"https://www.youtube.com/watch?v={video.youtube_id}"
                try:
                    info = ydl.extract_info(url, download=False)
                    if info:
                         upload_date_str = info.get('upload_date')
                         if upload_date_str:
                             video.published_at = datetime.strptime(upload_date_str, "%Y%m%d")
                             session.add(video)
                             session.commit()
                             print(f"  Updated: {video.published_at}")
                             count += 1
                except Exception as e:
                    print(f"  Error: {e}")
                    
        print(f"Backfill complete. Updated {count} videos.")

if __name__ == "__main__":
    backfill_dates()
