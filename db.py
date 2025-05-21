import sqlite3
from contextlib import contextmanager
import json
from pathlib import Path

DB_PATH = Path(__file__).with_name("cardio.db")

def init_db():
    with open(Path(__file__).with_name("migrations.sql")) as f:
        script = f.read()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(script)
    print("Database initialized ✅")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ---------- CRUD helpers ----------

def add_model(name, version, path):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO models (name, version, path) VALUES (?,?,?)",
            (name, version, path)
        )

def log_prediction(user_id, model_id, features_dict, output):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (user_id, model_id, input_json, output) "
            "VALUES (?,?,?,?)",
            (user_id, model_id, json.dumps(features_dict), int(output))
        )

def get_latest_model(name):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE name=? ORDER BY trained_at DESC LIMIT 1",
            (name,)
        ).fetchone()
        return row
