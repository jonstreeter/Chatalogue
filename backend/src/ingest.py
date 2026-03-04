import yt_dlp
import os
from pathlib import Path
from sqlmodel import Session, select
from .db.database import engine, Video, create_db_and_tables

# Configuration
CHANNEL_URL = "https://www.youtube.com/@example"
AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"

def ensure_dirs():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    create_db_and_tables()

def get_channel_videos():
    ydl_opts = {
        'extract_flat': True, # Don't download, just extract info
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(CHANNEL_URL, download=False)
        if 'entries' in result:
            return result['entries']
        return []

def download_audio(video_url: str, output_path: Path):
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': str(output_path),
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def ingest(limit: int = None):
    print("Ensuring directories and database...")
    ensure_dirs()
    
    print("Fetching video list...")
    videos_info = get_channel_videos()
    print(f"Found {len(videos_info)} videos.")
    
    if limit:
        print(f"Limiting to first {limit} videos.")
        videos_info = videos_info[:limit]

    with Session(engine) as session:
        for info in videos_info:
            # yt-dlp "flat" extraction has limited fields, might need to fetch more details 
            # or just rely on what we get for now. 
            # keys often: id, title, url, duration, upload_date...
            
            yt_id = info.get('id')
            if not yt_id: 
                continue
                
            # Check if exists
            statement = select(Video).where(Video.youtube_id == yt_id)
            existing = session.exec(statement).first()
            
            if not existing:
                print(f"New video found: {info.get('title')}")
                # Parse date if possible, or leave default/handle later. 
                # yt-dlp dates are 'YYYYMMDD' strings usually.
                from datetime import datetime
                upload_date_str = info.get('upload_date')
                pub_date = datetime.strptime(upload_date_str, "%Y%m%d") if upload_date_str else datetime.now()

                video = Video(
                    youtube_id=yt_id,
                    title=info.get('title', 'Unknown Title'),
                    published_at=pub_date,
                    duration=info.get('duration'),
                    thumbnail_url=info.get('thumbnails', [{}])[0].get('url') if info.get('thumbnails') else None
                )
                session.add(video)
                session.commit()
                session.refresh(video)
                existing = video
            
            # Download Check
            audio_path = AUDIO_DIR / f"{yt_id}.m4a"
            if not audio_path.exists():
                print(f"Downloading audio for {yt_id}...")
                try:
                    download_audio(f"https://www.youtube.com/watch?v={yt_id}", audio_path.with_suffix('')) # yt-dlp adds extension
                except Exception as e:
                    print(f"Failed to download {yt_id}: {e}")
            else:
                # print(f"Audio already exists for {yt_id}")
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    args = parser.parse_args()
    ingest(limit=args.limit)
