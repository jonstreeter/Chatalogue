import os
import time
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# Load env from backend root
# Assuming we are running from backend/ dir
load_dotenv(".env")

def download_file(url, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 1000:
        return
    print(f"Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}: Status {response.status_code}")
        # Create a dummy silent file if download fails using ffmpeg ? 
        # No, better to fail loud or use another source.
        # Fallback to local generation if possible (requires ffmpeg)
        import subprocess
        print("Generating dummy audio...")
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=10", "-c:a", "libvorbis", filename], check=False)

def benchmark(model_size, compute_type, audio_file):
    print(f"\n--- Benchmarking: {model_size} ({compute_type}) ---")
    
    start_load = time.time()
    try:
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    load_time = time.time() - start_load
    print(f"Model Load Time: {load_time:.2f}s")
    
    start_transcribe = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5)
    
    # Consume generator
    count = 0
    for _ in segments:
        count += 1
        
    transcribe_time = time.time() - start_transcribe
    print(f"Transcription Time: {transcribe_time:.2f}s")
    print(f"Audio Duration: {info.duration:.2f}s")
    print(f"Speedup Factor: {info.duration / transcribe_time:.2f}x")

def main():
    # JFK speech (11 sec)
    audio_url = "https://upload.wikimedia.org/wikipedia/commons/d/d4/En-us-jfk-inaugural-address-excerpt.ogg"
    audio_file = "jfk_test.ogg"
    download_file(audio_url, audio_file)

    configurations = [
        ("medium", "float16"),
        ("medium", "int8_float16"),
        ("large-v3", "float16"),
        ("large-v3", "int8_float16"),
        # Uncomment if you want to test distil (requires downloading distil model)
        # ("distil-whisper/distil-large-v3", "float16"),
        # ("distil-whisper/distil-large-v3", "int8_float16"),
    ]

    for size, compute in configurations:
        benchmark(size, compute, audio_file)

if __name__ == "__main__":
    if not os.path.exists(".env"):
        print("Warning: .env not found, using defaults")
    main()
