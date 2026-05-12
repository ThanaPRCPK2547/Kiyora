import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 1. LOAD DATA
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT_DIR / "dataset_extended_prepared.csv",
    ROOT_DIR / "data" / "dataset_extended_prepared.csv",
    Path("/mnt/user-data/uploads/dataset_extended_prepared.csv"),
]
data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("dataset_extended_prepared.csv not found.")

df = pd.read_csv(data_path)
print(f"[Load] {df.shape[0]} rows × {df.shape[1]} cols\n")

# 2. FEATURE ENGINEERING
df["province_encoded"] = (df["province_group"] == "Bangkok_Metropolitan").astype(int)
occ_dummies = pd.get_dummies(df["occupation_group"], prefix="occ").astype(int).iloc[:, 1:]
df = pd.concat([df, occ_dummies], axis=1)

dup_col = "prob_สิวเสี้ยน\u200b"
if dup_col in df.columns:
    df.drop(columns=[dup_col], inplace=True)

# Target A – Kiyora vs Others
df["target_kiyora"] = (
    df["brand_primary"].astype(str).str.strip().str.casefold() == "kiyora"
).astype(int)

prob_cols   = [c for c in df.columns if c.startswith("prob_")]
factor_cols = [c for c in df.columns if c.startswith("factor_")]
occ_cols    = list(occ_dummies.columns)

# 3. FEATURE SETS
# Model A – ใช้ทุก feature
FEAT_A = [
    *factor_cols,
    "age_encoded", "monthly_income_encoded", "acne_level_encoded",
    "skin_type_encoded", "cw_user_int", "province_encoded", "gender_female",
    *prob_cols,
    *occ_cols,
]
FEAT_A = [c for c in FEAT_A if c in df.columns]

# Model B1 – Acne: ยกเว้น acne_level_encoded
# Model B2 – Income: ยกเว้น monthly_income_encoded
base_cols = ["gender_female", "province_encoded", "skin_type_encoded",
             "cw_user_int", *occ_cols, *prob_cols]

FEAT_B1 = [c for c in base_cols + factor_cols + ["age_encoded"] if c in df.columns]
FEAT_B2 = [c for c in base_cols + factor_cols + ["age_encoded"] if c in df.columns]

# 4. LABEL GROUPING (Model B)
def group_acne(v):
    if v <= 0:   return 0
    elif v == 1: return 1
    elif v == 2: return 2
    else:        return 3

def group_income(v):
    if v <= 2:   return 0
    elif v <= 5: return 1
    else:        return 2

df["acne_group"]   = df["acne_level_encoded"].apply(group_acne)
df["income_group"] = df["monthly_income_encoded"].apply(group_income)

ACNE_LABELS   = {0: "ไม่มี/น้อยมาก", 1: "สิวเล็กน้อย", 2: "สิวปานกลาง", 3: "สิวรุนแรง"}
INCOME_LABELS = {0: "Low (<15k)", 1: "Mid (15-30k)", 2: "High (30k+)"}

# 5. SHARED HELPERS
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1),
        "SVM":                 SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42),
    }

def smote_lite(X, y, random_state=42):
    rng   = np.random.RandomState(random_state)
    X_min = X[y == 1]
    n_needed  = (y == 0).sum() - (y == 1).sum()
    synthetic = []
    for _ in range(n_needed):
        i, j  = rng.randint(0, len(X_min), 2)
        alpha = rng.rand()
        synthetic.append(X_min[i] + alpha * (X_min[j] - X_min[i]))
    return (np.vstack([X, np.vstack(synthetic)]),
            np.concatenate([y, np.ones(n_needed, dtype=int)]))

# 6. MODEL A – BINARY CLASSIFICATION (Kiyora vs Others)
print("=" * 65)
print("  MODEL A: ทำนายผู้ใช้ Kiyora (Binary Classification)")
print("=" * 65)

X_a = df[FEAT_A].fillna(0).values
y_a = df["target_kiyora"].values

print(f"Dataset: {len(y_a)} samples  |  Kiyora={y_a.sum()}  Others={(y_a==0).sum()}")
print(f"Features: {len(FEAT_A)}\n")

X_bal, y_bal = smote_lite(X_a, y_a)
print(f"After SMOTE-lite – Class 0: {(y_bal==0).sum()}  Class 1: {(y_bal==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
print(f"Train: {len(y_train)}  |  Test: {len(y_test)}\n")

results_a = {}
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV-F1':>7}")
print("=" * 65)

for name, clf in get_models().items():
    pipe  = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    results_a[name] = {
        "y_pred":    y_pred,
        "acc":       accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_test, y_prob),
        "cv_f1":     cv_f1,
    }
    r = results_a[name]
    print(f"{name:<22} {r['acc']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} "
          f"{r['f1']:>6.3f} {r['auc']:>6.3f} {r['cv_f1']:>7.3f}")

print("=" * 65)

best_a = max(results_a, key=lambda k: results_a[k]["cv_f1"])
r      = results_a[best_a]
print(f"\n★ Best Model (by CV F1): {best_a}")
print(f"   CV F1={r['cv_f1']:.3f}  |  AUC={r['auc']:.3f}  |  Accuracy={r['acc']:.3f}\n")

print("── Classification Report ──────────────────────────────────")
print(classification_report(y_test, r["y_pred"],
                             target_names=["Other Brand", "Kiyora"]))

print("── Confusion Matrix ────────────────────────────────────────")
cm = confusion_matrix(y_test, r["y_pred"])
print(f"                Predicted Other  Predicted Kiyora")
print(f"Actual Other         {cm[0,0]:>4}              {cm[0,1]:>4}")
print(f"Actual Kiyora        {cm[1,0]:>4}              {cm[1,1]:>4}")

# 7. MODEL B – CROSS-ANALYSIS (Acne Level × Income Group)
def train_evaluate_multi(X, y, label_map, task_name):
    print(f"\n{'='*65}")
    print(f"  {task_name}")
    print(f"{'='*65}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train={len(y_train)}  |  Test={len(y_test)}\n")

    results = {}
    print(f"{'Model':<22} {'Acc':>6} {'F1-Macro':>9} {'CV-F1':>7}")
    print("-" * 47)

    for name, clf in get_models().items():
        pipe  = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv,
                                scoring="f1_macro").mean()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            "pipe":     pipe,
            "y_pred":   y_pred,
            "acc":      accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "cv_f1":    cv_f1,
        }
        r = results[name]
        print(f"{name:<22} {r['acc']:>6.3f} {r['f1_macro']:>9.3f} {r['cv_f1']:>7.3f}")

    best_name   = max(results, key=lambda k: results[k]["cv_f1"])
    best        = results[best_name]
    class_names = [label_map[k] for k in sorted(label_map.keys())]
    print(f"\n★ Best: {best_name}  (CV F1={best['cv_f1']:.3f})")

    print("\n── Classification Report ──────────────────────────────────")
    print(classification_report(y_test, best["y_pred"],
                                 target_names=class_names, zero_division=0))

    rf_clf = results["Random Forest"]["pipe"].named_steps["clf"]
    feat_names = X.columns.tolist() if hasattr(X, "columns") else list(range(X.shape[1]))
    fi = pd.DataFrame({"feature": feat_names,
                       "importance": rf_clf.feature_importances_})\
           .sort_values("importance", ascending=False)
    print("── Top 10 Features (Random Forest) ───────────────────────")
    print(fi.head(10).to_string(index=False))

    return results[best_name]["pipe"], best_name

print("\n\n" + "=" * 65)
print("  MODEL B: Cross-Analysis – Acne Level × Income Group")
print("=" * 65)

print("\n[Acne groups]")
for k, v in ACNE_LABELS.items():
    print(f"  {k} = {v}: {(df['acne_group']==k).sum()} คน")

print("\n[Income groups]")
for k, v in INCOME_LABELS.items():
    print(f"  {k} = {v}: {(df['income_group']==k).sum()} คน")

X_b1 = df[FEAT_B1].fillna(0)
y_b1 = df["acne_group"].values
pipe_acne, best_acne = train_evaluate_multi(
    X_b1, y_b1, ACNE_LABELS, "MODEL B1: ทำนาย Acne Level")

X_b2 = df[FEAT_B2].fillna(0)
y_b2 = df["income_group"].values
pipe_income, best_income = train_evaluate_multi(
    X_b2, y_b2, INCOME_LABELS, "MODEL B2: ทำนาย Income Group")

# Cross-Analysis
df["pred_acne_label"]   = pipe_acne.predict(X_b1.values)
df["pred_income_label"] = pipe_income.predict(X_b2.values)
df["pred_acne_label"]   = df["pred_acne_label"].map(ACNE_LABELS)
df["pred_income_label"] = df["pred_income_label"].map(INCOME_LABELS)
df["is_kiyora"]         = df["target_kiyora"]

print(f"\n{'='*65}")
print("  CROSS-ANALYSIS: Acne Level × Income Group → Segment")
print(f"{'='*65}")

seg_all    = df.groupby(["pred_acne_label", "pred_income_label"]).size().reset_index(name="total")
seg_kiyora = df[df["is_kiyora"] == 1]\
               .groupby(["pred_acne_label", "pred_income_label"])\
               .size().reset_index(name="kiyora_users")
seg = seg_all.merge(seg_kiyora, on=["pred_acne_label", "pred_income_label"], how="left").fillna(0)
seg["kiyora_users"] = seg["kiyora_users"].astype(int)
seg["kiyora_%"]     = (seg["kiyora_users"] / seg["total"] * 100).round(1)
seg = seg.sort_values("kiyora_users", ascending=False)
seg.columns = ["Acne Level", "Income Group", "Total", "Kiyora Users", "Kiyora %"]
seg.index = range(1, len(seg) + 1)

print("\n── Segment Table (เรียงตาม Kiyora Users) ─────────────────")
print(seg.to_string())

print("\n── Pivot: จำนวน Kiyora Users ต่อ Segment ─────────────────")
pivot_k = df[df["is_kiyora"] == 1].pivot_table(
    index="pred_acne_label", columns="pred_income_label",
    values="is_kiyora", aggfunc="sum", fill_value=0)
print(pivot_k.to_string())

print("\n── Pivot: สัดส่วน Kiyora ใน Segment (%) ──────────────────")
pivot_total = df.pivot_table(
    index="pred_acne_label", columns="pred_income_label",
    values="is_kiyora", aggfunc="count", fill_value=0)
pivot_pct = (pivot_k / pivot_total * 100).round(1).fillna(0)
print(pivot_pct.to_string())

print(f"\n[Model A Best]      {best_a}")
print(f"[Model B1 Best]     {best_acne}")
print(f"[Model B2 Best]     {best_income}")

# SAVE RESULTS TO JSON
MODEL_DIR = Path(__file__).resolve().parent

# Model A Results
sup_results = {
    "model_a": {
        "best_model": best_a,
        "metrics": {
            "cv_f1": float(results_a[best_a]["cv_f1"]),
            "auc": float(results_a[best_a]["auc"]),
            "accuracy": float(results_a[best_a]["acc"]),
            "precision": float(results_a[best_a]["precision"]),
            "recall": float(results_a[best_a]["recall"]),
            "f1": float(results_a[best_a]["f1"]),
        },
        "confusion_matrix": {
            "actual_other": {"predicted_other": int(cm[0,0]), "predicted_kiyora": int(cm[0,1])},
            "actual_kiyora": {"predicted_other": int(cm[1,0]), "predicted_kiyora": int(cm[1,1])},
        },
        "class_balance": [
            {"label": "Other Brand", "value": int((y_a == 0).sum())},
            {"label": "Kiyora", "value": int((y_a == 1).sum())},
        ],
    },
    "model_b": {
        "best_acne_model": best_acne,
        "best_income_model": best_income,
        "acne_labels": ACNE_LABELS,
        "income_labels": INCOME_LABELS,
        "segments": seg.to_dict(orient="records"),
    },
    "summary": {
        "total_samples": int(len(df)),
        "kiyora_users": int(df["target_kiyora"].sum()),
        "kiyora_share": float(df["target_kiyora"].mean() * 100),
    },
}

# Save to JSON
output_path = MODEL_DIR / "sup_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sup_results, f, ensure_ascii=False, indent=2)

print(f"\n✓ Results saved to: {output_path}")
