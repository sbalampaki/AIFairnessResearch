import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MODEL_CANDIDATES = []

try:
    from xgboost import XGBClassifier
    MODEL_CANDIDATES.append(("XGBoost", lambda: XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1
    )))
except Exception:
    pass

# Always include these two (installed with scikit-learn)
MODEL_CANDIDATES.append(("RandomForest", lambda: RandomForestClassifier(
    n_estimators=500, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1
)))
MODEL_CANDIDATES.append(("LogisticRegression", lambda: LogisticRegression(
    max_iter=2000, class_weight="balanced", n_jobs=None
)))

# Keep only 3 models as requested (prioritize stronger ones if available)
MODEL_CANDIDATES = MODEL_CANDIDATES[:3]

# -----------------------------
# Helper functions for metrics
# -----------------------------
def _rates(y_true, y_pred):
    # Returns TPR, FPR, P(ŷ=1)
    # handle cases where confusion_matrix might not see both labels
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape != (2, 2):
        # pad to 2x2 if needed
        padded = np.zeros((2, 2), dtype=int)
        idx = {lab: i for i, lab in enumerate(sorted(np.unique(y_true)))}
        for i_true, lab_true in enumerate(sorted(np.unique(y_true))):
            for i_pred, lab_pred in enumerate(sorted(np.unique(y_pred))):
                padded[idx[lab_true], idx[lab_pred]] = cm[i_true, i_pred]
        tn, fp, fn, tp = padded.ravel()
    else:
        tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    p_pred_pos = (tp + fp) / max(tn + fp + fn + tp, 1)
    return tpr, fpr, p_pred_pos

def _safe_ratio(a, b):
    # min/max ratio (robust)
    arr = np.array([a, b], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return np.nan
    mx = np.max(arr)
    if mx == 0:
        return np.nan
    return np.min(arr) / mx

def demographic_parity(y_pred_g0, y_pred_g1, y_true_g0=None, y_true_g1=None):
    # parity uses predicted positive rates only
    _, _, ppos0 = _rates(y_true_g0 if y_true_g0 is not None else np.zeros_like(y_pred_g0), y_pred_g0)
    _, _, ppos1 = _rates(y_true_g1 if y_true_g1 is not None else np.zeros_like(y_pred_g1), y_pred_g1)
    return (ppos1 - ppos0, _safe_ratio(ppos0, ppos1))

def equal_opportunity(y_true_g0, y_pred_g0, y_true_g1, y_pred_g1):
    tpr0, _, _ = _rates(y_true_g0, y_pred_g0)
    tpr1, _, _ = _rates(y_true_g1, y_pred_g1)
    return (tpr1 - tpr0, _safe_ratio(tpr0, tpr1))

def equalized_odds(y_true_g0, y_pred_g0, y_true_g1, y_pred_g1):
    tpr0, fpr0, _ = _rates(y_true_g0, y_pred_g0)
    tpr1, fpr1, _ = _rates(y_true_g1, y_pred_g1)
    tpr_diff = tpr1 - tpr0
    fpr_diff = fpr1 - fpr0
    eo_agg_diff = np.nanmax(np.abs([tpr_diff, fpr_diff]))
    tpr_ratio = _safe_ratio(tpr0, tpr1)
    fpr_ratio = _safe_ratio(fpr0, fpr1)
    return {
        "eo_tpr_diff": tpr_diff,
        "eo_fpr_diff": fpr_diff,
        "eo_agg_diff": eo_agg_diff,
        "eo_tpr_ratio": tpr_ratio,
        "eo_fpr_ratio": fpr_ratio,
    }

# -----------------------------
# Paths / config
# -----------------------------
splits = [1, 2, 3, 4, 5]
data_dir = "data"
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

summary_rows = []

for i in splits:
    train_path = os.path.join(data_dir, f"germancredit_split{i}_train.csv")
    test_path  = os.path.join(data_dir, f"germancredit_split{i}_test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"[Split {i}] Missing files, skipping.")
        continue

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Columns to exclude from model features
    sensitive_cols = ["gender", "age", "race"]
    target_col = "target"
    feature_cols = [c for c in df_train.columns if c not in sensitive_cols + [target_col]]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test[target_col].values

    # Gender masks on TEST set
    g_series = df_test["gender"].astype(str).str.lower()
    mask_f = g_series.eq("female")
    mask_m = g_series.eq("male")

    # Prepare per-split results
    split_rows = []

    for model_name, ctor in MODEL_CANDIDATES:
        try:
            clf = ctor()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        except Exception as e:
            # If a model fails (e.g., missing package), record NaNs
            print(f"[Split {i}] {model_name} failed: {e}")
            res = {
                "split": i, "model": model_name, "accuracy": np.nan,
                "dp_diff_f_minus_m": np.nan, "dp_ratio_min_over_max": np.nan,
                "eopp_tpr_diff_f_minus_m": np.nan, "eopp_tpr_ratio_min_over_max": np.nan,
                "eodds_tpr_diff_f_minus_m": np.nan, "eodds_fpr_diff_f_minus_m": np.nan,
                "eodds_agg_max_abs_gap": np.nan,
                "eodds_tpr_ratio_min_over_max": np.nan, "eodds_fpr_ratio_min_over_max": np.nan,
                "n_female": int(mask_f.sum()), "n_male": int(mask_m.sum())
            }
            split_rows.append(res)
            continue

        # Slice into groups (handle missing group gracefully)
        def _subset(y_t, y_p, m):
            return y_t[m], y_p[m]

        if mask_f.any():
            y_true_f, y_pred_f = _subset(y_test, y_pred, mask_f)
        else:
            y_true_f, y_pred_f = np.array([], dtype=int), np.array([], dtype=int)

        if mask_m.any():
            y_true_m, y_pred_m = _subset(y_test, y_pred, mask_m)
        else:
            y_true_m, y_pred_m = np.array([], dtype=int), np.array([], dtype=int)

        # Ensure non-empty for metric functions
        def _nan_if_empty(arr):
            return arr if arr.size else np.array([np.nan])

        y_true_f = _nan_if_empty(y_true_f)
        y_pred_f = _nan_if_empty(y_pred_f)
        y_true_m = _nan_if_empty(y_true_m)
        y_pred_m = _nan_if_empty(y_pred_m)

        # Metrics
        dp_diff, dp_ratio = demographic_parity(y_pred_m, y_pred_f, y_true_m, y_true_f)
        eopp_diff, eopp_ratio = equal_opportunity(y_true_m, y_pred_m, y_true_f, y_pred_f)
        eodds = equalized_odds(y_true_m, y_pred_m, y_true_f, y_pred_f)

        res = {
            "split": i,
            "model": model_name,
            "accuracy": acc,
            # Demographic Parity (positive prediction rate)
            "dp_diff_f_minus_m": dp_diff,                # P(ŷ=1|F) - P(ŷ=1|M)
            "dp_ratio_min_over_max": dp_ratio,           # min/max across groups
            # Equal Opportunity (TPR parity)
            "eopp_tpr_diff_f_minus_m": eopp_diff,        # TPR_F - TPR_M
            "eopp_tpr_ratio_min_over_max": eopp_ratio,
            # Equalized Odds (TPR & FPR parity)
            "eodds_tpr_diff_f_minus_m": eodds["eo_tpr_diff"],
            "eodds_fpr_diff_f_minus_m": eodds["eo_fpr_diff"],
            "eodds_agg_max_abs_gap": eodds["eo_agg_diff"],      # max(|ΔTPR|, |ΔFPR|)
            "eodds_tpr_ratio_min_over_max": eodds["eo_tpr_ratio"],
            "eodds_fpr_ratio_min_over_max": eodds["eo_fpr_ratio"],
            # Diagnostics
            "n_female": int(mask_f.sum()),
            "n_male": int(mask_m.sum()),
        }
        split_rows.append(res)

    # Save per-split CSV (one row per model)
    split_df = pd.DataFrame(split_rows)
    split_csv = os.path.join(out_dir, f"fairness_gender_split{i}.csv")
    split_df.to_csv(split_csv, index=False)
    print(f"[Split {i}] Saved {split_csv}")

    summary_rows.extend(split_rows)

# Save combined summary across all splits/models
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(out_dir, "fairness_gender_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nCombined summary saved to {summary_csv}")

print("\nModels used this run:", [name for name, _ in MODEL_CANDIDATES])
