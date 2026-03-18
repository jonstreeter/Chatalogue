import psycopg
import json
conn = psycopg.connect("postgresql://chatalogue@127.0.0.1:55432/chatalogue")
conn.execute("INSERT INTO job (video_id, job_type, status, progress, payload_json, created_at) VALUES (3464, 'process', 'queued', 0, '{}', CURRENT_TIMESTAMP)")
conn.commit()
print("Job queued manually via psycopg")
