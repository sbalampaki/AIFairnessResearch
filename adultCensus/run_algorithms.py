
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate


MODELS = [
    ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ("RandomForest", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)),
]
try:
    from xgboost import XGBClassifier
    MODELS.append(("XGBoost", XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        eval_metric="logloss", random_state=42, n_jobs=-1
    )))
    print("Using XGBoost as well.")
except Exception:
    print("xgboost not installed; running without it.")

# ---------------------------
# Helper to compute summaries
# ---------------------------
def fairness_summaries(y_true, y_pred, sensitive):
    """Return dict of DP/EOpp/EOdds summaries using MetricFrame."""
    mf_tpr = MetricFrame(metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    mf_fpr = MetricFrame(metrics=false_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    mf_sr  = MetricFrame(metrics=selection_rate,    y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)

    # Demographic Parity (selection rate parity)
    dp_diff  = mf_sr.difference()
    dp_ratio = mf_sr.ratio()

    # Equal Opportunity (TPR parity)
    eopp_diff  = mf_tpr.difference()
    eopp_ratio = mf_tpr.ratio()

    # Equalized Odds (TPR & FPR parity)
    eodds_tpr_diff  = mf_tpr.difference()
    eodds_fpr_diff  = mf_fpr.difference()
    eodds_tpr_ratio = mf_tpr.ratio()
    eodds_fpr_ratio = mf_fpr.ratio()
    eodds_agg_max_abs_gap = np.nanmax(np.abs([eodds_tpr_diff, eodds_fpr_diff]))

    return {
        "dp_diff": dp_diff,
        "dp_ratio": dp_ratio,
        "eopp_diff": eopp_diff,
        "eopp_ratio": eopp_ratio,
        "eodds_tpr_diff": eodds_tpr_diff,
        "eodds_fpr_diff": eodds_fpr_diff,
        "eodds_agg_max_abs_gap": eodds_agg_max_abs_gap,
        "eodds_tpr_ratio": eodds_tpr_ratio,
        "eodds_fpr_ratio": eodds_fpr_ratio,
    }

def by_group_table(y_true, y_pred, sensitive):
    """Return a long/flattened DataFrame of by-group metrics."""
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    # Convert to a tidy table: one row per group with all metrics
    df_bg = mf.by_group.reset_index()
    df_bg.rename(columns={"index": "group"}, inplace=True)
    if "group" not in df_bg.columns:  # newer fairlearn keeps the sensitive col name as the index name
        df_bg.rename(columns={df_bg.columns[0]: "group"}, inplace=True)
    return df_bg

# ---------------------------
# I/O config
# ---------------------------
splits = [1, 2, 3, 4, 5]
data_dir = "data"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

sensitive_features = ["race", "gender", "education"]

for split in splits:
    train_path = os.path.join(data_dir, f"adult_split{split}_train.csv")
    test_path  = os.path.join(data_dir, f"adult_split{split}_test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"[Split {split}] Missing files; skipping.")
        continue

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    target = "target"
    feature_cols = [c for c in df_train.columns if c not in sensitive_features + [target]]

    X_train, y_train = df_train[feature_cols].values, df_train[target].values
    X_test,  y_test  = df_test[feature_cols].values,  df_test[target].values

    for sf in sensitive_features:
        rows_summary = []        # one row per model (overall & fairness summaries)
        rows_by_group = []       # one row per (model × group) with per-group metrics

        sens_test = df_test[sf]

        for model_name, model in MODELS:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Overall accuracy
            acc = accuracy_score(y_test, y_pred)

            # Fairness summaries
            fsum = fairness_summaries(y_test, y_pred, sens_test)
            row_sum = {"split": split, "sensitive_feature": sf, "model": model_name, "accuracy": acc}
            row_sum.update(fsum)
            rows_summary.append(row_sum)

            # Per-group metrics (flattened)
            df_bg = by_group_table(y_test, y_pred, sens_test)
            df_bg.insert(0, "split", split)
            df_bg.insert(1, "sensitive_feature", sf)
            df_bg.insert(2, "model", model_name)
            rows_by_group.append(df_bg)

        # Combine into a single CSV per split × sensitive_feature
        df_summary = pd.DataFrame(rows_summary)

        if rows_by_group:
            df_groups = pd.concat(rows_by_group, ignore_index=True)
            # Merge summaries and group metrics in one file (different sections)
            # We'll add a column 'section' to distinguish
            df_summary["_section"] = "summary"
            df_groups["_section"] = "by_group"
            # Align columns (outer join-like union)
            common_cols = list(set(df_summary.columns).union(df_groups.columns))
            df_out = pd.concat(
                [df_summary.reindex(columns=common_cols), df_groups.reindex(columns=common_cols)],
                ignore_index=True
            )
        else:
            df_out = df_summary
            df_out["_section"] = "summary"

        out_path = os.path.join(results_dir, f"split{split}_{sf}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"[Split {split}] saved {out_path}")
