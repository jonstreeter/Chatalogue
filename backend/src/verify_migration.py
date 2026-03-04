import os
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# EXTENSIVE MOCKING
# Mock all missing dependencies in sys.modules
mock_modules = [
    'yt_dlp', 
    'sqlmodel', 
    'dotenv', 
    'numpy', 
    'faster_whisper', 
    'pyannote', 
    'pyannote.audio', 
    'pyannote.core',
    'scipy',
    'scipy.spatial.distance',
    'torch'
]

for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# Determine paths
current_dir = Path(__file__).parent.resolve() # backend/src
backend_dir = current_dir.parent # backend
sys.path.append(str(backend_dir))

# Now we can import ingestion
try:
    from src.services import ingestion
    # We need to monkeypath the classes Video, Channel, Job because they are imported from db.database
    # but db.database imports sqlmodel which is mocked, so Video/Channel are Mocks.
    # We can just work with Mocks.
    
    # Reload to ensure mocks are used
    import importlib
    importlib.reload(ingestion)
    
    from src.services.ingestion import IngestionService
except ImportError as e:
    print(f"Import Error: {e}")
    # Inspecting where it failed
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify the logic
# logic uses: Session(engine), session.get(Channel, ...), video.channel_id, video.title
# logic uses: AUDIO_DIR (from ingestion module)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
AUDIO_DIR = DATA_DIR / "audio"

# We need to overwrite ingestion.AUDIO_DIR to point to our test area if it isn't relative to __file__ correctly in mock env
# In ingestion.py: AUDIO_DIR = DATA_DIR / "audio"
# DATA_DIR = Path(__file__)... 
# Should be correct relative to the file location.

# Mock the session context manager
mock_session = MagicMock()
mock_session.__enter__.return_value = mock_session

# We also need to patch Session in ingestion module.
# In ingestion.py: from sqlmodel import Session
# Since sqlmodel is a Mock, Session is a Mock.
# We can configure that Mock.
ingestion.Session.side_effect = lambda engine: mock_session

# Mock methods
video_id = 1
yt_id = "test_vid_123"
channel_id = 1
channel_name = "Test Channel"
video_title = "Test Episode"

# Setup DB returns
# session.get(Channel, channel_id) -> channel obj
mock_channel = MagicMock()
mock_channel.name = channel_name
mock_channel.id = channel_id

mock_video = MagicMock()
mock_video.youtube_id = yt_id
mock_video.channel_id = channel_id
mock_video.title = video_title

def get_side_effect(model, id):
    # In ingestion.py: session.get(Channel, video.channel_id)
    # We don't know exactly what 'Channel' class is (it's a mock).
    # But we can check if it matches the one passed.
    # Actually, simpler: just return based on some condition or always return mock_channel if it looks like a channel request
    return mock_channel

mock_session.get.side_effect = get_side_effect

print("Initializing IngestionService...")
# ingestion.create_db_and_tables is a Mock, so it's fine.
service = IngestionService()

print("Setting up test files...")
expected_dir = AUDIO_DIR / "TestChannel" / "TestEpisode"
# Clean up
if expected_dir.parent.exists():
    shutil.rmtree(expected_dir.parent)

old_file = AUDIO_DIR / f"{yt_id}.m4a"
old_file.parent.mkdir(parents=True, exist_ok=True)
with open(old_file, "w") as f:
    f.write("test content")

print("Running download_audio...")
# Call the method
# Note: download_audio uses 'Video' type hint, but at runtime python 3.10+ doesn't enforce it.
# We pass mock_video.
try:
    with patch.object(service, '_update_job_progress'): # avoid errors
        new_path = service.download_audio(mock_video)
except Exception as e:
    print(f"Error executing download_audio: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check results
print(f"Returned path: {new_path}")
expected_file = expected_dir / "TestEpisode.m4a"

if expected_file.exists():
    print("SUCCESS: New file exists.")
else:
    print(f"FAILURE: New file not found at {expected_file}")

if not old_file.exists():
    print("SUCCESS: Old file removed.")
else:
    print("FAILURE: Old file still exists.")

# Clean up
if expected_dir.parent.exists():
    shutil.rmtree(expected_dir.parent)
