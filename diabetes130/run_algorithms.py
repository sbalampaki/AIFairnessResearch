# If needed:
# !pip install fairlearn xgboost

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate

# Optional XGBoost
MODELS = [
    ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ("RandomForest", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                            random_state=42, n_jobs=-1)),
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

# ---- metric helpers ----
def fairness_summaries(y_true, y_pred, sensitive):
    mf_tpr = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    mf_fpr = MetricFrame(
        metrics=false_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    mf_sr  = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )

    return {
        "dp_diff": mf_sr.difference(),
        "dp_ratio": mf_sr.ratio(),
        "eopp_diff": mf_tpr.difference(),
        "eopp_ratio": mf_tpr.ratio(),
        "eodds_tpr_diff": mf_tpr.difference(),
        "eodds_fpr_diff": mf_fpr.difference(),
        "eodds_agg_max_abs_gap": np.nanmax(
            np.abs([mf_tpr.difference(), mf_fpr.difference()])
        ),
        "eodds_tpr_ratio": mf_tpr.ratio(),
        "eodds_fpr_ratio": mf_fpr.ratio(),
    }

def by_group_table(y_true, y_pred, sensitive):
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
    df = mf.by_group.reset_index()
    if df.columns[0] != "group":
        df.rename(columns={df.columns[0]: "group"}, inplace=True)
    return df

# ---- config ----
splits = [1, 2, 3, 4, 5]
data_dir = "data"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

sensitive_features = ["medicare", "medicaid", "age"]

for split in splits:
    train_path = os.path.join(data_dir, f"diab_hosp_split{split}_train.csv")
    test_path  = os.path.join(data_dir, f"diab_hosp_split{split}_test.csv")
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
        s_test = df_test[sf]

        # --- summaries: one row per model ---
        rows_summary = []
        # --- by-group: one row per (model × group) ---
        rows_bygroup = []

        for model_name, model in MODELS:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_overall = accuracy_score(y_test, y_pred)

            # Summaries
            fsum = fairness_summaries(y_test, y_pred, s_test)
            rows_summary.append({
                "split": split,
                "sensitive_feature": sf,
                "model": model_name,
                "accuracy": acc_overall,
                **fsum
            })

            # By-group
            df_bg = by_group_table(y_test, y_pred, s_test)
            df_bg.insert(0, "split", split)
            df_bg.insert(1, "sensitive_feature", sf)
            df_bg.insert(2, "model", model_name)
            rows_bygroup.append(df_bg)

        # Save two separate files per split × sensitive feature
        out_summary = pd.DataFrame(rows_summary)
        out_bygroup = pd.concat(rows_bygroup, ignore_index=True) if rows_bygroup else pd.DataFrame(
            columns=["split", "sensitive_feature", "model", "group", "accuracy", "tpr", "fpr", "selection_rate"]
        )

        path_summary = os.path.join(results_dir, f"diab_split{split}_{sf}_summary.csv")
        path_bygroup = os.path.join(results_dir, f"diab_split{split}_{sf}_bygroup.csv")
        out_summary.to_csv(path_summary, index=False)
        out_bygroup.to_csv(path_bygroup, index=False)

        print(f"[Split {split}] saved {path_summary}")
        print(f"[Split {split}] saved {path_bygroup}")