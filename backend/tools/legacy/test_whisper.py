from faster_whisper import WhisperModel
import time

print("Starting WhisperModel test...")
start = time.time()
try:
    model = WhisperModel("tiny", device="cpu", compute_type="float32")
    print(f"Model loaded in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error loading model: {e}")
