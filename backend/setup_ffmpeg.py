import os
import shutil
import zipfile
import urllib.request
from pathlib import Path
import sys

# URL for the latest shared release build from BtbN (includes DLLs)
# We need the "shared" version for torchcodec/torchaudio to function correctly.
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
BACKEND_DIR = Path(__file__).parent
BIN_DIR = BACKEND_DIR / "bin"
TEMP_ZIP = BACKEND_DIR / "ffmpeg_temp.zip"

def setup_ffmpeg():
    print("Setting up local FFmpeg (Shared Build)...")
    
    # Create bin directory
    if not BIN_DIR.exists():
        BIN_DIR.mkdir(exist_ok=True)
        print(f"Created {BIN_DIR}")
        
    ffmpeg_exe = BIN_DIR / "ffmpeg.exe"
    ffprobe_exe = BIN_DIR / "ffprobe.exe"
    # Check for a key DLL to see if we have the shared version
    avcodec_dll = BIN_DIR / "avcodec-61.dll" # Version might change, but presence of any dll is good check?
    # Actually, let's just force update if user runs this script, or check if ffmpeg exists.
    # But since we are switching versions, maybe we should force?
    # For now, let's check if ffmpeg exists.
    
    if ffmpeg_exe.exists() and ffprobe_exe.exists():
        print("FFmpeg executables found.")
        # Optional: could check for DLLs to decide if we need to upgrade from static to shared
        # But let's assume if the user runs this, they want to setup/update.
        # return 

    print(f"Downloading FFmpeg from {FFMPEG_URL}...")
    try:
        # Download with progress
        def reporthook(blocknum, blocksize, totalsize):
            readso = blocknum * blocksize
            if totalsize > 0:
                percent = readso * 100 / totalsize
                s = "\rDownload progress: %5.1f%% %*d / %d" % (
                    percent, len(str(totalsize)), readso, totalsize)
                sys.stdout.write(s)
                if readso >= totalsize: # near the end
                    sys.stdout.write("\n")
        
        # Add headers to avoid 403 Forbidden from GitHub (sometimes needed)
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(FFMPEG_URL, TEMP_ZIP, reporthook=reporthook)
        print("Download complete.")
        
        print("Extracting...")
        with zipfile.ZipFile(TEMP_ZIP, 'r') as zip_ref:
            # The zip contains a root folder like 'ffmpeg-master-latest-win64-gpl-shared/'
            # We want to extract everything from the 'bin/' subfolder inside it.
            
            for file in zip_ref.namelist():
                # Check if it is inside a 'bin/' folder
                # Example: ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe
                # Example: ffmpeg-master-latest-win64-gpl-shared/bin/avcodec-61.dll
                if "/bin/" in file and not file.endswith("/"):
                    filename = Path(file).name
                    target_path = BIN_DIR / filename
                    
                    print(f"Extracting {filename}...")
                    source = zip_ref.open(file)
                    target = open(target_path, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
                        
    except Exception as e:
        print(f"Error installing FFmpeg: {e}")
        # Clean up partials?
    finally:
        if TEMP_ZIP.exists():
            os.remove(TEMP_ZIP)
            print("Cleaned up temp zip.")

    print(f"FFmpeg setup complete. Binaries and DLLs located in {BIN_DIR}")

if __name__ == "__main__":
    setup_ffmpeg()
