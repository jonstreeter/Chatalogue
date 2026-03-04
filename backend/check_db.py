import sys
import os
sys.path.append(os.path.abspath("src"))
from db.database import engine
from sqlmodel import Session, text

with Session(engine) as session:
    res = session.execute(text("SELECT id, status, job_type FROM job ORDER BY id DESC LIMIT 5;")).fetchall()
    for r in res:
        print(r)
