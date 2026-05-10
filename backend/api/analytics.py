from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.api.supabase_client import (
    SupabaseConfigError,
    get_supabase,
    get_supabase_table,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "model"

FACTOR_LABELS = {
    "factor_deep_cleansing": "Deep cleansing",
    "factor_acne_friendly": "Acne friendly",
    "factor_sensitive_friendly": "Sensitive friendly",
    "factor_hypoallergenic": "Hypoallergenic",
    "factor_moisturizing": "Moisturizing",
    "factor_low_friction": "Low friction",
    "factor_nourishment": "Nourishment",
    "factor_eye_friendly": "Eye friendly",
    "factor_oil_control": "Oil control",
    "factor_no_allergen": "No allergen",
}

DEMOGRAPHIC_FEATURES = [
    "age_encoded",
    "monthly_income_encoded",
    "skin_type_encoded",
    "acne_level_encoded",
    "uses_cleansing_water",
]

CLUSTER_NAMES = {
    0: "Everyday Gentle Users",
    1: "Passive Legacy Users",
    2: "Sensitive Acne Care Seekers",
}


def _load_sup_results() -> dict | None:
    path = MODEL_DIR / "sup_results.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_cluster_profile() -> "pd.DataFrame | None":
    for path in (MODEL_DIR / "kiyora_cluster_profile.csv", ROOT_DIR / "kiyora_cluster_profile.csv"):
        if path.exists():
            try:
                return pd.read_csv(path, index_col=0)
            except Exception:
                pass
    return None


def _load_clustered_df() -> "pd.DataFrame | None":
    for path in (MODEL_DIR / "kiyora_clustered.csv", ROOT_DIR / "kiyora_clustered.csv"):
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    return None


@lru_cache(maxsize=1)
def build_overview() -> dict[str, Any]:
    df, source = _load_data()
    if df.empty:
        return {
            "source": source,
            "summary": {
                "rows": 0,
                "columns": 0,
                "kiyora_users": 0,
                "kiyora_share": 0,
                "cleansing_water_share": 0,
            },
            "supervised": {},
            "unsupervised": {},
        }

    df = _coerce_numeric(df)
    brand = df.get("brand_primary", pd.Series(dtype=str)).astype(str).str.strip()
    kiyora_mask = brand.str.casefold() == "kiyora"

    factor_cols = [col for col in FACTOR_LABELS if col in df.columns]
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    cluster_features = [
        col for col in DEMOGRAPHIC_FEATURES + factor_cols + prob_cols if col in df.columns
    ]

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "kiyora_users": int(kiyora_mask.sum()),
        "kiyora_share": _pct(kiyora_mask.mean()),
        "cleansing_water_share": _pct(_safe_mean(df, "uses_cleansing_water")),
        "source_label": source,
    }

    return {
        "source": source,
        "summary": summary,
        "supervised": _build_supervised(df, brand, kiyora_mask, factor_cols, prob_cols),
        "unsupervised": _build_unsupervised(df, brand, kiyora_mask, cluster_features),
    }


def _load_data() -> tuple[pd.DataFrame, str]:
    supabase_df = _load_supabase_data()
    if _has_analysis_columns(supabase_df):
        return supabase_df, f"Supabase:{get_supabase_table()}"

    for path in (
        ROOT_DIR / "dataset_extended_prepared.csv",
        ROOT_DIR / "data" / "dataset_extended_prepared.csv",
        Path("/mnt/user-data/uploads/dataset_extended_prepared.csv"),
    ):
        if path.exists():
            return pd.read_csv(path), f"CSV:{path.name}"

    if not supabase_df.empty:
        return supabase_df, f"Supabase:{get_supabase_table()}"
    return pd.DataFrame(), "No data"


def _load_supabase_data() -> pd.DataFrame:
    try:
        response = get_supabase().table(get_supabase_table()).select("*").limit(2000).execute()
    except Exception:
        return pd.DataFrame()

    rows = response.data or []
    normalized = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("data"), dict):
            normalized.append({**row["data"], "_id": row.get("id")})
        elif isinstance(row, dict):
            normalized.append(row)
    return pd.DataFrame(normalized)


def _has_analysis_columns(df: pd.DataFrame) -> bool:
    return not df.empty and {"brand_primary", "age_encoded"}.issubset(df.columns)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [*FACTOR_LABELS, *DEMOGRAPHIC_FEATURES]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_supervised(
    df: pd.DataFrame,
    brand: pd.Series,
    kiyora_mask: pd.Series,
    factor_cols: list[str],
    prob_cols: list[str],
) -> dict[str, Any]:
    top_brands = _top_counts(brand, limit=8)
    factors = _factor_means(df, factor_cols)
    candidate_features = [
        col
        for col in [
            *factor_cols,
            "age_encoded",
            "monthly_income_encoded",
            "acne_level_encoded",
            "skin_type_encoded",
            "uses_cleansing_water",
            "gender_female",
            *prob_cols,
        ]
        if col in df.columns
    ]

    sup = _load_sup_results()
    if sup:
        model_a = sup.get("model_a", {})
        best_model = model_a.get("best_model", "Random Forest")
        class_balance = model_a.get("class_balance") or [
            {"label": "Other Brand", "value": int((~kiyora_mask).sum())},
            {"label": "Kiyora", "value": int(kiyora_mask.sum())},
        ]
    else:
        best_model = "Random Forest"
        class_balance = [
            {"label": "Other Brand", "value": int((~kiyora_mask).sum())},
            {"label": "Kiyora", "value": int(kiyora_mask.sum())},
        ]

    result: dict[str, Any] = {
        "target": "Kiyora vs Other Brand",
        "best_model": best_model,
        "feature_count": len(candidate_features),
        "class_balance": class_balance,
        "brand_mix": top_brands,
        "factor_scores": factors,
        "feature_importance": _mutual_information(df, kiyora_mask, candidate_features),
        "model_results_available": sup is not None,
    }

    if sup:
        model_a = sup.get("model_a", {})
        if model_a.get("metrics"):
            result["model_metrics"] = model_a["metrics"]
        if model_a.get("confusion_matrix"):
            result["confusion_matrix"] = model_a["confusion_matrix"]
        if sup.get("model_b"):
            result["model_b"] = sup["model_b"]

    return result


def _build_unsupervised(
    df: pd.DataFrame,
    brand: pd.Series,
    kiyora_mask: pd.Series,
    cluster_features: list[str],
) -> dict[str, Any]:
    if len(df) < 4 or len(cluster_features) < 2:
        return {
            "cluster_count": 0,
            "features": len(cluster_features),
            "clusters": [],
            "silhouette": [],
            "kiyora_cluster_mix": [],
        }

    x = df[cluster_features].fillna(0)
    x_scaled = StandardScaler().fit_transform(x)
    cluster_count = min(4, max(2, len(df) - 1))
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(x_scaled)

    work = df.copy()
    work["cluster"] = labels
    clusters = []
    for cluster_id, cluster_df in work.groupby("cluster"):
        cluster_brand = brand.loc[cluster_df.index]
        cluster_kiyora = kiyora_mask.loc[cluster_df.index]
        top_brand = cluster_brand.value_counts().index[0] if not cluster_brand.empty else "Unknown"
        clusters.append(
            {
                "id": int(cluster_id),
                "size": int(len(cluster_df)),
                "share": _pct(len(cluster_df) / len(df)),
                "kiyora_share": _pct(cluster_kiyora.mean()),
                "top_brand": str(top_brand),
                "avg_age": _round(cluster_df["age_encoded"].mean())
                if "age_encoded" in cluster_df
                else None,
                "avg_income": _round(cluster_df["monthly_income_encoded"].mean())
                if "monthly_income_encoded" in cluster_df
                else None,
            }
        )

    result: dict[str, Any] = {
        "cluster_count": cluster_count,
        "features": len(cluster_features),
        "clusters": sorted(clusters, key=lambda row: row["id"]),
        "silhouette": _silhouette_scores(x_scaled),
        "kiyora_cluster_mix": _cluster_mix(labels, kiyora_mask),
    }

    kiyora_clusters = _build_kiyora_clusters()
    if kiyora_clusters:
        result["kiyora_clusters"] = kiyora_clusters

    return result


def _build_kiyora_clusters() -> list[dict[str, Any]] | None:
    clustered = _load_clustered_df()
    profile = _load_cluster_profile()
    if clustered is None or "cluster" not in clustered.columns:
        return None

    factor_cols_p = [c for c in FACTOR_LABELS if profile is not None and c in profile.columns]
    clusters = []
    for cluster_id, cluster_df in clustered.groupby("cluster"):
        info: dict[str, Any] = {
            "id": int(cluster_id),
            "name": CLUSTER_NAMES.get(int(cluster_id), f"Cluster {cluster_id}"),
            "size": int(len(cluster_df)),
        }
        if profile is not None and int(cluster_id) in profile.index:
            row = profile.loc[int(cluster_id)]
            info["acne_level"] = _round(row.get("acne_level_encoded", 0))
            info["skin_type"] = _round(row.get("skin_type_encoded", 0))
            if factor_cols_p and row[factor_cols_p].max() > 0:
                top_col = row[factor_cols_p].idxmax()
                info["top_factor"] = FACTOR_LABELS.get(top_col, top_col)
                info["factor_scores"] = [
                    {"label": FACTOR_LABELS[c], "value": _round(row[c])}
                    for c in factor_cols_p
                ]
        clusters.append(info)
    return sorted(clusters, key=lambda r: r["id"])


def get_supervised_results() -> dict[str, Any]:
    path = MODEL_DIR / "sup_results.json"
    if not path.exists():
        return {"available": False}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True
        return data
    except Exception:
        return {"available": False}


def get_unsupervised_results() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cluster_names": {str(k): v for k, v in CLUSTER_NAMES.items()},
    }
    kiyora_clusters = _build_kiyora_clusters()
    if kiyora_clusters:
        result["kiyora_clusters"] = kiyora_clusters
        result["available"] = True
    else:
        result["available"] = False

    profile = _load_cluster_profile()
    if profile is not None:
        factor_cols_p = [c for c in FACTOR_LABELS if c in profile.columns]
        profiles = []
        for cluster_id in sorted(profile.index):
            row = profile.loc[cluster_id]
            profiles.append({
                "cluster": int(cluster_id),
                "name": CLUSTER_NAMES.get(int(cluster_id), f"Cluster {cluster_id}"),
                "acne_level": _round(row.get("acne_level_encoded", 0)),
                "skin_type": _round(row.get("skin_type_encoded", 0)),
                "factors": {
                    FACTOR_LABELS[c]: _round(row[c])
                    for c in factor_cols_p
                    if not pd.isna(row[c])
                },
            })
        result["profile"] = profiles

    return result


def _mutual_information(
    df: pd.DataFrame, kiyora_mask: pd.Series, candidate_features: list[str]
) -> list[dict[str, Any]]:
    if not candidate_features or kiyora_mask.nunique() < 2:
        return []

    x = df[candidate_features].fillna(0)
    y = kiyora_mask.astype(int).to_numpy()
    scores = mutual_info_classif(x, y, random_state=42, discrete_features="auto")
    rows = []
    for feature, score in zip(candidate_features, scores):
        rows.append({"label": _feature_label(feature), "value": _round(score)})
    return sorted(rows, key=lambda row: row["value"], reverse=True)[:8]


def _silhouette_scores(x_scaled: np.ndarray) -> list[dict[str, Any]]:
    scores = []
    max_k = min(7, len(x_scaled) - 1)
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x_scaled)
        scores.append({"k": k, "score": _round(silhouette_score(x_scaled, labels))})
    return scores


def _cluster_mix(labels: np.ndarray, kiyora_mask: pd.Series) -> list[dict[str, Any]]:
    rows = []
    labels_series = pd.Series(labels, index=kiyora_mask.index)
    for cluster_id, group in labels_series.groupby(labels_series):
        mask = labels_series.index.isin(group.index)
        rows.append(
            {
                "label": f"Cluster {int(cluster_id)}",
                "value": int(kiyora_mask.loc[mask].sum()),
            }
        )
    return rows


def _factor_means(df: pd.DataFrame, factor_cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for col in factor_cols:
        value = df[col].replace(0, np.nan).mean()
        rows.append({"label": FACTOR_LABELS[col], "value": _round(value)})
    return sorted(rows, key=lambda row: row["value"], reverse=True)


def _top_counts(series: pd.Series, limit: int) -> list[dict[str, Any]]:
    return [
        {"label": str(label), "value": int(value)}
        for label, value in series.replace("nan", "Unknown").value_counts().head(limit).items()
    ]


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())


def _feature_label(feature: str) -> str:
    if feature in FACTOR_LABELS:
        return FACTOR_LABELS[feature]
    return feature.replace("prob_", "").replace("_", " ")


def _round(value: Any, digits: int = 3) -> float:
    if value is None or pd.isna(value):
        return 0
    return round(float(value), digits)


def _pct(value: Any) -> float:
    return _round(float(value) * 100 if not pd.isna(value) else 0, 1)
