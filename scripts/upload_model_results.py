"""
Upload trained model results to Supabase.

Usage:
    python scripts/upload_model_results.py

Requires SUPABASE_URL and SUPABASE_KEY in .env (or environment).
Run once after training; re-running safely upserts the latest results.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

try:
    from supabase import create_client
except ImportError:
    print("ERROR: supabase package not installed. Run: pip install supabase")
    sys.exit(1)

import pandas as pd

MODEL_DIR = ROOT_DIR / "model"
TABLE = "model_results"


def get_client():
    url = os.getenv("SUPABASE_URL")
    # Prefer service role key (bypasses RLS) for uploads; fall back to anon key.
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        sys.exit(1)
    return create_client(url, key)


def upsert(client, result_type: str, payload: dict):
    client.table(TABLE).upsert(
        {"result_type": result_type, "payload": payload},
        on_conflict="result_type",
    ).execute()
    print(f"  [OK] {result_type}")


def upload_sup_results(client):
    path = MODEL_DIR / "sup_results.json"
    if not path.exists():
        print(f"  [SKIP] sup_results.json not found at {path}")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    upsert(client, "sup_results", data)


def upload_cluster_profile(client):
    path = MODEL_DIR / "kiyora_cluster_profile.csv"
    if not path.exists():
        print(f"  [SKIP] kiyora_cluster_profile.csv not found at {path}")
        return
    df = pd.read_csv(path, index_col=0)
    records = df.reset_index().rename(columns={"index": "cluster"}).to_dict(orient="records")
    upsert(client, "cluster_profile", {"profiles": records})


def upload_clustered(client):
    path = MODEL_DIR / "kiyora_clustered.csv"
    if not path.exists():
        print(f"  [SKIP] kiyora_clustered.csv not found at {path}")
        return
    df = pd.read_csv(path)
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    upsert(client, "clustered", {"rows": records})


def main():
    print(f"Connecting to Supabase table '{TABLE}' ...")
    client = get_client()
    print("Uploading model results:")
    upload_sup_results(client)
    upload_cluster_profile(client)
    upload_clustered(client)
    print("Done.")


if __name__ == "__main__":
    main()
