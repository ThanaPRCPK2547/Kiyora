import os
from functools import lru_cache
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
else:
    load_dotenv()


class SupabaseConfigError(RuntimeError):
    pass


def get_supabase_table() -> str:
    return os.getenv("SUPABASE_TABLE", "kiyora_records")


@lru_cache(maxsize=1)
def get_supabase() -> Any:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if (
        not url
        or not key
        or "your-project-ref" in url
        or key == "your-supabase-server-key"
    ):
        raise SupabaseConfigError(
            "Set SUPABASE_URL and SUPABASE_KEY before using Supabase endpoints."
        )

    try:
        from supabase import create_client
    except ImportError as exc:
        raise SupabaseConfigError(
            "Install the supabase package with `pip install -r requirements.txt`."
        ) from exc

    return create_client(url, key)
