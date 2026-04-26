"""
AIE323 - Kiyora Clean Supervised Learning: Binary Classification
Target : target_kiyora (1 = uses Kiyora, 0 = uses other brands)
Models : Logistic Regression, Random Forest, SVM, KNN, Decision Tree
Handle imbalance via SMOTE-lite + class_weight='balanced'
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT_DIR / "data" / "dataset_extended_prepared.csv",
    ROOT_DIR / "kiyora_ml_ready.csv",
]

data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        f"Dataset not found. Expected one of: {', '.join(str(path) for path in DATA_CANDIDATES)}"
    )

df = pd.read_csv(data_path)
if "target_kiyora" not in df.columns:
    if "brand_primary" not in df.columns:
        raise KeyError("Expected either 'target_kiyora' or 'brand_primary' in the dataset.")
    df["target_kiyora"] = (
        df["brand_primary"].astype(str).str.strip().str.casefold() == "kiyora"
    ).astype(int)

FEATURE_COLS = [
    "factor_deep_cleansing",
    "factor_acne_friendly",
    "factor_sensitive_friendly",
    "factor_no_allergen",
    "factor_hypoallergenic",
    "factor_moisturizing",
    "factor_low_friction",
    "factor_nourishment",
    "factor_eye_friendly",
    "factor_oil_control",
    "age_encoded",
    "monthly_income_encoded",
    "acne_level_encoded",
    "skin_type_encoded",
    "cw_user_int",
    "province_encoded",
    "gender_female",
    "prob_uneven skintone",
    "prob_à¸œà¸´à¸§à¸¡à¸±à¸™à¹€à¸¢à¸´à¹‰à¸¡",
    "prob_à¸œà¸´à¸§à¸«à¸¡à¸­à¸‡à¸„à¸¥à¹‰à¸³",
    "prob_à¸œà¸´à¸§à¹à¸žà¹‰à¸‡à¹ˆà¸²à¸¢",
    "prob_à¸œà¸´à¸§à¹à¸«à¹‰à¸‡/à¸¥à¸­à¸/à¹€à¸›à¹‡à¸™à¸‚à¸¸à¸¢",
    "prob_à¸à¹‰à¸² à¸à¸£à¸°",
    "prob_à¸£à¸­à¸¢à¸ªà¸´à¸§",
    "prob_à¸£à¸´à¹‰à¸§à¸£à¸­à¸¢",
    "prob_à¸£à¸¹à¸‚à¸¸à¸¡à¸‚à¸™à¸à¸§à¹‰à¸²à¸‡",
    "prob_à¸ªà¸´à¸§à¸œà¸”",
    "prob_à¸ªà¸´à¸§à¸­à¸±à¸à¹€à¸ªà¸š",
    "prob_à¸ªà¸´à¸§à¸­à¸¸à¸”à¸•à¸±à¸™",
    "prob_à¸ªà¸´à¸§à¹€à¸ªà¸µà¹‰à¸¢à¸™",
    "prob_à¸«à¸¥à¸¸à¸¡à¸ªà¸´à¸§",
    "prob_à¹„à¸¡à¹ˆà¸¡à¸µà¸›à¸±à¸à¸«à¸²à¸œà¸´à¸§",
    "occ_Gov_Employee",
    "occ_Other",
    "occ_Private_Employee",
    "occ_Student",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS].fillna(0).values
y = df["target_kiyora"].values

print(f"Dataset: {len(y)} samples  |  Kiyora={y.sum()}  Others={(y==0).sum()}")
print(f"Features: {len(FEATURE_COLS)}\n")


# ==============================================================================
# 2. HANDLE IMBALANCE - SMOTE-lite (manual oversampling)
# ==============================================================================
def smote_lite(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    X_min = X[y == 1]
    n_needed = (y == 0).sum() - (y == 1).sum()
    synthetic = []
    for _ in range(n_needed):
        i, j = rng.randint(0, len(X_min), 2)
        alpha = rng.rand()
        synthetic.append(X_min[i] + alpha * (X_min[j] - X_min[i]))
    X_bal = np.vstack([X, np.vstack(synthetic)])
    y_bal = np.concatenate([y, np.ones(n_needed, dtype=int)])
    return X_bal, y_bal


X_bal, y_bal = smote_lite(X, y)
print(f"After SMOTE-lite - Class 0: {(y_bal==0).sum()}  Class 1: {(y_bal==1).sum()}")


# ==============================================================================
# 3. TRAIN / TEST SPLIT  (stratified, test on balanced data)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"Train: {len(y_train)}  |  Test: {len(y_test)}\n")


# ==============================================================================
# 4. DEFINE MODELS
# ==============================================================================
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "SVM": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
}


# ==============================================================================
# 5. TRAIN, CROSS-VALIDATE & EVALUATE
# ==============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=" * 65)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV-F1':>7}")
print("=" * 65)

for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "pipe": pipe,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "acc": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "cv_f1": cv_f1,
        "cv_auc": cv_auc,
    }
    r = results[name]
    print(
        f"{name:<22} {r['acc']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} "
        f"{r['f1']:>6.3f} {r['auc']:>6.3f} {r['cv_f1']:>7.3f}"
    )

print("=" * 65)


# ==============================================================================
# 6. BEST MODEL REPORT
# ==============================================================================
best_name = max(results, key=lambda k: results[k]["cv_f1"])
best = results[best_name]

print(f"\nBest Model (by CV F1): {best_name}")
print(f"   CV F1={best['cv_f1']:.3f}  |  AUC={best['auc']:.3f}  |  Accuracy={best['acc']:.3f}\n")

print("Classification Report")
print(classification_report(y_test, best["y_pred"], target_names=["Other Brand", "Kiyora"]))

print("Confusion Matrix")
cm = confusion_matrix(y_test, best["y_pred"])
print("                Predicted Other  Predicted Kiyora")
print(f"Actual Other         {cm[0,0]:>4}              {cm[0,1]:>4}")
print(f"Actual Kiyora        {cm[1,0]:>4}              {cm[1,1]:>4}")


# ==============================================================================
# 7. FEATURE IMPORTANCE (Random Forest)
# ==============================================================================
rf_pipe = results["Random Forest"]["pipe"]
rf_clf = rf_pipe.named_steps["clf"]
fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": rf_clf.feature_importances_}).sort_values(
    "importance", ascending=False
)

print("\nTop 10 Features (Random Forest Importance)")
print(fi.head(10).to_string(index=False))
