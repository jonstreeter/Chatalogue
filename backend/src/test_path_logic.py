import re
import shutil
from pathlib import Path

def sanitize_filename(name: str) -> str:
    """Sanitize string to be filesystem safe"""
    # Replace invalid characters for Windows/Linux
    # Remove invalid chars: < > : " / \ | ? *
    s = re.sub(r'[<>:"/\\|?*]', '', name)
    # Strip leading/trailing spaces and dots
    s = s.strip().strip('.')
    return s or "Unknown"

def test_path_logic():
    print("Testing sanitize_filename...")
    assert sanitize_filename("Normal Name") == "Normal Name"
    assert sanitize_filename("Name: With / Bad <Chars>") == "Name With  Bad Chars"
    assert sanitize_filename("   Spaces   ") == "Spaces"
    assert sanitize_filename("...") == "Unknown"
    print("Sanitization OK.")

    # Test file migration logic (mocked)
    AUDIO_DIR = Path("test_audio_data")
    if AUDIO_DIR.exists():
        shutil.rmtree(AUDIO_DIR)
    AUDIO_DIR.mkdir()

    # Inputs
    channel_name = "My Channel / VLOG"
    video_title = "Episode 1: The Beginning?"
    yt_id = "video123"

    safe_channel = sanitize_filename(channel_name)
    safe_title = sanitize_filename(video_title)
    
    print(f"Safe Channel: '{safe_channel}'")
    print(f"Safe Title: '{safe_title}'")

    episode_dir = AUDIO_DIR / safe_channel / safe_title
    expected_path = episode_dir / f"{safe_title}.m4a"

    # Scenario 1: Old file exists
    old_path = AUDIO_DIR / f"{yt_id}.m4a"
    old_path.touch()
    
    # Simulate logic
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    if expected_path.exists():
        print("New path already exists.")
    elif old_path.exists():
        print(f"Migrating {old_path} -> {expected_path}")
        old_path.rename(expected_path)
    
    assert expected_path.exists()
    assert not old_path.exists()
    print("Migration OK.")

    # Cleanup
    shutil.rmtree(AUDIO_DIR)

if __name__ == "__main__":
    test_path_logic()
