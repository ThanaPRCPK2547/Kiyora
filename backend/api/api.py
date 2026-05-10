from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.analytics import build_overview, get_supervised_results, get_unsupervised_results
from backend.api.supabase_client import (
    SupabaseConfigError,
    get_supabase,
    get_supabase_table,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
PUBLIC_DIR = ROOT_DIR / "public"

app = FastAPI(title="Kiyora Dashboard API")

class Item(BaseModel):
    name: str
    price: float
    quantity: int

class RecordPayload(BaseModel):
    data: dict[str, Any] | list[dict[str, Any]] = Field(
        ...,
        description="One row or a list of rows to insert into the configured Supabase table.",
    )

@app.get("/", include_in_schema=False)
def read_root():
    index_path = PUBLIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return read_api_info()

@app.get("/api")
def read_api_info():
    return {
        "service": "Kiyora API",
        "database": "supabase",
        "table": get_supabase_table(),
    }

@app.get("/health")
@app.get("/api/health")
def read_health():
    return {
        "status": "ok",
        "supabase_configured": _supabase_is_configured(),
        "table": get_supabase_table(),
    }

@app.get("/api/overview")
def read_overview(refresh: bool = Query(default=False)):
    if refresh:
        build_overview.cache_clear()
    return build_overview()


@app.get("/api/model/supervised")
def read_supervised_model():
    data = get_supervised_results()
    if not data.get("available"):
        raise HTTPException(
            status_code=404,
            detail="Supervised model results not found. Run model/sup.py first.",
        )
    return data


@app.get("/api/model/unsupervised")
def read_unsupervised_model():
    return get_unsupervised_results()

@app.get("/records")
@app.get("/api/records")
def list_records(limit: int = Query(default=100, ge=1, le=1000)):
    supabase = _get_supabase_or_503()
    response = supabase.table(get_supabase_table()).select("*").limit(limit).execute()
    return {"data": response.data}

@app.post("/records", status_code=201)
@app.post("/api/records", status_code=201)
def insert_records(payload: RecordPayload):
    supabase = _get_supabase_or_503()
    source_rows = payload.data if isinstance(payload.data, list) else [payload.data]
    rows = [{"data": row} for row in source_rows]
    response = supabase.table(get_supabase_table()).insert(rows).execute()
    return {"data": response.data}

def _supabase_is_configured() -> bool:
    try:
        get_supabase()
    except SupabaseConfigError:
        return False
    return True

def _get_supabase_or_503():
    try:
        return get_supabase()
    except SupabaseConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
