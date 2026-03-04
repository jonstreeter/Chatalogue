import requests
import time

def test_jobs_endpoint():
    try:
        start = time.time()
        print("Sending GET request to http://localhost:8000/jobs...")
        response = requests.get("http://localhost:8000/jobs", timeout=5)
        duration = time.time() - start
        
        print(f"Status Code: {response.status_code}")
        print(f"Time Taken: {duration:.4f}s")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            print("API seems healthy.")
        else:
            print("API returned error.")
            
    except requests.exceptions.Timeout:
        print("request timed out! API is hanging.")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_jobs_endpoint()
