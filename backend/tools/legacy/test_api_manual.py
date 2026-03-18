import urllib.request
import json
import time
import urllib.parse
import sys

BASE_URL = "http://localhost:8000"

def get_json(url):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"GET {url} failed: {e}")
        return None

def post_json(url, params=None):
    try:
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"POST {url} failed: {e}")
        raise

print(f"Testing API at {BASE_URL}...")
try:
    # 1. GET Channels
    print("\n1. Listing Channels...")
    channels = get_json(f"{BASE_URL}/channels")
    print(f"Current channels: {channels}")
    
    # 2. Add Channel
    target_url = "https://www.youtube.com/@example"
    print(f"\n2. Adding channel {target_url}...")
    
    # Check if already exists to avoid error or duplicate logic (though DB handles it)
    exists = next((c for c in (channels or []) if c['url'] == target_url), None)
    if exists:
        print("Channel already exists.")
        cid = exists['id']
    else:
        channel = post_json(f"{BASE_URL}/channels", {"url": target_url})
        print("Channel Added:", channel)
        cid = channel['id']
    
    # 3. Refresh
    print(f"\n3. Triggering Refresh for Channel ID {cid}...")
    try:
        res = post_json(f"{BASE_URL}/channels/{cid}/refresh")
        print("Refresh response:", res)
    except Exception as e:
        print("Refresh trigger failed (might be busy?):", e)
    
    # 4. Wait and List Videos
    print("\n4. Waiting 10s for ingestion background task...")
    time.sleep(10)
    videos = get_json(f"{BASE_URL}/videos")
    print(f"Total Videos found: {len(videos) if videos else 0}")
    if videos:
        print(f"First video: {videos[0]['title']}")

    # 5. Search Test
    print("\n5. Testing Search (if implemented)...")
    try:
        # Search for a common word like "the" just to see if it returns anything (if validation data exists)
        # Note: Search depends on TranscriptSegments which won't exist until Transcription finishes.
        # Transcription is slow on CPU/Start.
        print("Skipping search test as transcription takes time.")
    except:
        pass

    print("\nTest Complete.")

except Exception as e:
    print(f"\nCRITICAL FAILURE: {e}")
    sys.exit(1)
