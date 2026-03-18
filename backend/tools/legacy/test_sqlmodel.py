import sys
import os
sys.path.append(os.path.abspath('src'))
from db.database import Job
j = Job(video_id=1, job_type="process", error="No module named 'pkg_resources'\nTraceback:...")
print("REPR OF ERROR:", repr(j.error))
