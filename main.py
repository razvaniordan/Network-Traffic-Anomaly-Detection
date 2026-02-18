import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


warnings.filterwarnings("ignore")


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    train_csv: str = "UNSW_NB15_training-set.csv"
    test_csv: str = "UNSW_NB15_testing-set.csv"
    out_dir: str = "outputs"
    seed: int = 42

    # Unsupervised thresholds
    iforest_contamination: float = 0.02  # used by IsolationForest for internal calibration
    threshold_quantile_on_normal: float = 0.98  # choose threshold from normal scores (98th percentile)

    # Supervised
    use_smote: bool = True

    # Autoencoder
    ae_epochs: int = 15
    ae_batch_size: int = 1024
    ae_threshold_quantile_on_normal: float = 0.98


CFG = Config()


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    os.makedirs(os.path.join(CFG.out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(CFG.out_dir, "metrics"), exist_ok=True)


def save_fig(name: str):
    path = os.path.join(CFG.out_dir, "figures", name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[FIG] Saved: {path}")


def plot_histograms(df: pd.DataFrame, numeric_cols, max_cols: int = 12):
    cols = numeric_cols[:max_cols]
    df[cols].hist(bins=50, figsize=(16, 10))
    plt.suptitle("Feature Distributions (Numeric) - Histograms")
    save_fig("histograms_numeric.png")


def plot_boxplots(df: pd.DataFrame, numeric_cols, max_cols: int = 12):
    cols = numeric_cols[:max_cols]
    plt.figure(figsize=(16, 8))
    df[cols].plot(kind="box", vert=False)
    plt.title("Feature Distributions (Numeric) - Boxplots")
    save_fig("boxplots_numeric.png")

def plot_boxplots_after_scaling(df: pd.DataFrame, numeric_cols, output_path):
    """
    Plots boxplots of numeric features AFTER standard scaling.
    This shows feature distributions as seen by ML models.
    """

    # 1. Extract numeric data
    X = df[numeric_cols]

    # 2. Scale features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Create boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        X_scaled,
        vert=False,
        labels=numeric_cols,
        showfliers=False
    )

    plt.title("Feature Distributions (Numeric) - Boxplots After Scaling")
    plt.xlabel("Standardized value (z-score)")
    plt.tight_layout()

    # 4. Save figure
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[FIG] Saved: {output_path}")



def plot_corr_heatmap(df: pd.DataFrame, numeric_cols, max_cols: int = 25):
    cols = numeric_cols[:max_cols]
    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr.values, aspect="auto")
    plt.title("Correlation Heatmap (subset of numeric features)")
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    plt.yticks(range(len(cols)), cols, fontsize=7)
    plt.colorbar()
    save_fig("correlation_heatmap_subset.png")


def summarize_dataset(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    summary = {
        "name": name,
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values_total": int(df.isna().sum().sum()),
        "label_distribution": df["label"].value_counts().to_dict() if "label" in df.columns else None,
        "attack_cat_distribution": df["attack_cat"].value_counts().to_dict() if "attack_cat" in df.columns else None,
        "describe_numeric": df.describe(include=[np.number]).T.reset_index().to_dict(orient="records"),
    }
    return summary


def eval_binary(y_true, y_pred, y_score=None) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred).tolist()
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    out = {
        "confusion_matrix": cm,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    # support is None when using average="binary", so compute it manually
    out["support"] = int(np.sum(y_true == 1))

    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = None

    return out



# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(CFG.train_csv)
    test = pd.read_csv(CFG.test_csv)

    # Often, 'id' is not predictive; we drop it.
    for df in (train, test):
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    return train, test


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    # Identify categorical columns (proto/service/state are key categorical features in UNSW-NB15)
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # One-hot encode categoricals, scale numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop"
    )
    return preprocessor, categorical_cols, numeric_cols


# ----------------------------
# Part 2: Unsupervised Anomaly Detection (Isolation Forest)
# ----------------------------
def run_isolation_forest(train_df: pd.DataFrame, test_df: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[str, Any]:
    # Train on normal only
    train_normal = train_df[train_df["label"] == 0].copy()

    X_train = train_normal.drop(columns=["label", "attack_cat"], errors="ignore")
    X_test = test_df.drop(columns=["label", "attack_cat"], errors="ignore")
    y_test = test_df["label"].astype(int).values

    # Fit preprocessor on training data
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    iforest = IsolationForest(
        n_estimators=300,
        contamination=CFG.iforest_contamination,
        random_state=CFG.seed,
        n_jobs=-1
    )
    iforest.fit(X_train_t)

    # IsolationForest: decision_function -> higher = more normal; score_samples -> higher = more normal
    normality_score_train = iforest.decision_function(X_train_t)
    normality_score_test = iforest.decision_function(X_test_t)

    # Convert to anomaly score (higher = more anomalous)
    anomaly_score_train = -normality_score_train
    anomaly_score_test = -normality_score_test

    # Threshold from normal training distribution (quantile)
    thr = float(np.quantile(anomaly_score_train, CFG.threshold_quantile_on_normal))
    y_pred = (anomaly_score_test >= thr).astype(int)

    metrics = eval_binary(y_test, y_pred, y_score=anomaly_score_test)
    metrics.update({
        "threshold": thr,
        "threshold_quantile_on_normal": CFG.threshold_quantile_on_normal
    })

    # Plot anomaly score distribution
    plt.figure(figsize=(10, 5))
    plt.hist(anomaly_score_test[y_test == 0], bins=60, alpha=0.7, label="Normal (test)")
    plt.hist(anomaly_score_test[y_test == 1], bins=60, alpha=0.7, label="Attack (test)")
    plt.axvline(thr, linestyle="--", linewidth=2, label=f"Threshold={thr:.3f}")
    plt.title("Isolation Forest - Anomaly Score Distribution (Test)")
    plt.xlabel("Anomaly score (higher = more anomalous)")
    plt.ylabel("Count")
    plt.legend()
    save_fig("iforest_anomaly_scores.png")

    return {"model": "IsolationForest", "metrics": metrics}


# ----------------------------
# Part 3: Supervised Intrusion Classification (Binary)
# ----------------------------
def run_supervised_models(train_df: pd.DataFrame, test_df: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[str, Any]:
    X_train = train_df.drop(columns=["label", "attack_cat"], errors="ignore")
    y_train = train_df["label"].astype(int)

    X_test = test_df.drop(columns=["label", "attack_cat"], errors="ignore")
    y_test = test_df["label"].astype(int).values

    # Transform
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Optional SMOTE (only on training)
    if CFG.use_smote:
        smote = SMOTE(random_state=CFG.seed)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_t, y_train)
    else:
        X_train_bal, y_train_bal = X_train_t, y_train

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, n_jobs=-1)
    lr.fit(X_train_bal, y_train_bal)
    lr_score = lr.predict_proba(X_test_t)[:, 1]
    lr_pred = (lr_score >= 0.5).astype(int)
    results["LogisticRegression"] = eval_binary(y_test, lr_pred, y_score=lr_score)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=CFG.seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf.fit(X_train_t, y_train)  # RF can handle imbalance well; we keep original distribution
    rf_score = rf.predict_proba(X_test_t)[:, 1]
    rf_pred = (rf_score >= 0.5).astype(int)
    results["RandomForest"] = eval_binary(y_test, rf_pred, y_score=rf_score)

    # Save a bar plot comparison
    names = list(results.keys())
    f1s = [results[n]["f1"] for n in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, f1s)
    plt.title("Supervised Models (Binary) - F1 Comparison")
    plt.ylabel("F1-score")
    save_fig("supervised_binary_f1.png")

    return {"task": "SupervisedBinary", "metrics_by_model": results}


# ----------------------------
# Part 4: Deep Learning Autoencoder Anomaly Detection
# ----------------------------
def build_autoencoder(input_dim: int) -> keras.Model:
    # Simple dense AE (good baseline)
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def run_autoencoder(train_df: pd.DataFrame, test_df: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[str, Any]:
    # Train on normal only
    train_normal = train_df[train_df["label"] == 0].copy()

    X_train = train_normal.drop(columns=["label", "attack_cat"], errors="ignore")
    X_test = test_df.drop(columns=["label", "attack_cat"], errors="ignore")
    y_test = test_df["label"].astype(int).values

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # AE expects float32
    X_train_t = X_train_t.astype("float32")
    X_test_t = X_test_t.astype("float32")

    ae = build_autoencoder(X_train_t.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    ae.fit(
        X_train_t, X_train_t,
        validation_split=0.1,
        epochs=CFG.ae_epochs,
        batch_size=CFG.ae_batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # Reconstruction error
    recon_train = ae.predict(X_train_t, batch_size=CFG.ae_batch_size, verbose=0)
    recon_test = ae.predict(X_test_t, batch_size=CFG.ae_batch_size, verbose=0)

    train_err = np.mean((X_train_t - recon_train) ** 2, axis=1)
    test_err = np.mean((X_test_t - recon_test) ** 2, axis=1)

    thr = float(np.quantile(train_err, CFG.ae_threshold_quantile_on_normal))
    y_pred = (test_err >= thr).astype(int)

    metrics = eval_binary(y_test, y_pred, y_score=test_err)
    metrics.update({
        "threshold": thr,
        "threshold_quantile_on_normal": CFG.ae_threshold_quantile_on_normal
    })

    # Plot reconstruction error
    plt.figure(figsize=(10, 5))
    plt.hist(test_err[y_test == 0], bins=60, alpha=0.7, label="Normal (test)")
    plt.hist(test_err[y_test == 1], bins=60, alpha=0.7, label="Attack (test)")
    plt.axvline(thr, linestyle="--", linewidth=2, label=f"Threshold={thr:.6f}")
    plt.title("Autoencoder - Reconstruction Error (Test)")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Count")
    plt.legend()
    save_fig("autoencoder_recon_error.png")

    plt.figure(figsize=(10, 5))
    plt.hist(test_err[y_test == 0], bins=60, alpha=0.7, label="Normal (test)")
    plt.hist(test_err[y_test == 1], bins=60, alpha=0.7, label="Attack (test)")
    plt.axvline(thr, linestyle="--", linewidth=2, label=f"Threshold={thr:.6f}")
    plt.title("Autoencoder - Reconstruction Error (Test)")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Count")
    plt.legend()
    plt.xlim(0, 0.2)
    save_fig("autoencoder_recon_error_zoomed.png")

    return {"model": "Autoencoder", "metrics": metrics}


# ----------------------------
# Main (Part 1 + Parts 2-4 + Part 5 summary)
# ----------------------------
def main():
    ensure_dirs()

    train_df, test_df = load_data()

    # ---------------- Part 1: Exploration ----------------
    X_for_schema = train_df.drop(columns=["label", "attack_cat"], errors="ignore")
    preprocessor, cat_cols, num_cols = build_preprocessor(X_for_schema)

    print("\n=== Dataset Info ===")
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    print(f"Categorical cols: {cat_cols}")
    print(f"Numeric cols count: {len(num_cols)}")
    print(f"Missing values (train): {train_df.isna().sum().sum()} | (test): {test_df.isna().sum().sum()}")

    # label distribution
    print("\nTrain label distribution:\n", train_df["label"].value_counts())
    print("\nTrain attack_cat distribution (top):\n", train_df["attack_cat"].value_counts().head(12))

    # Summary stats + plots
    plot_histograms(train_df, num_cols)
    plot_boxplots(train_df, num_cols)
    plot_boxplots_after_scaling(train_df, num_cols, "outputs/figures/boxplots_numeric_scaled.png")
    plot_corr_heatmap(train_df, num_cols)

    # Save exploration summary
    exploration = {
        "train": summarize_dataset(train_df, "train"),
        "test": summarize_dataset(test_df, "test"),
        "categorical_columns": cat_cols,
        "numeric_columns_count": len(num_cols),
        "notes": {
            "missing_values": "No NaN values detected in provided CSVs. Some categoricals may use '-' to indicate 'no service'.",
            "class_imbalance": "Binary label is imbalanced; attacks are majority in training set, and attack categories are highly skewed (e.g., Worms rare)."
        }
    }

    # ---------------- Part 2: Unsupervised ----------------
    print("\n=== Running Unsupervised: Isolation Forest (train on normal only) ===")
    iforest_result = run_isolation_forest(train_df, test_df, preprocessor)

    # ---------------- Part 3: Supervised ----------------
    print("\n=== Running Supervised: Logistic Regression + Random Forest (binary) ===")
    supervised_result = run_supervised_models(train_df, test_df, preprocessor)

    # ---------------- Part 4: Autoencoder ----------------
    print("\n=== Running Deep Learning: Autoencoder (train on normal only) ===")
    ae_result = run_autoencoder(train_df, test_df, preprocessor)

    # ---------------- Part 5: Comparison ----------------
    comparison = {
        "unsupervised_iforest": iforest_result,
        "supervised_binary": supervised_result,
        "autoencoder": ae_result
    }

    # Simple summary table in console
    print("\n=== QUICK SUMMARY (Binary) ===")
    print("IsolationForest    F1:", comparison["unsupervised_iforest"]["metrics"]["f1"])
    print("Autoencoder        F1:", comparison["autoencoder"]["metrics"]["f1"])
    for m, vals in comparison["supervised_binary"]["metrics_by_model"].items():
        print(f"{m:>16}   F1:", vals["f1"])

    # Save all metrics to JSON
    out = {
        "exploration": exploration,
        "results": comparison,
        "config": CFG.__dict__
    }
    metrics_path = os.path.join(CFG.out_dir, "metrics", "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] Saved metrics JSON: {metrics_path}")
    print("[OK] All done.")


if __name__ == "__main__":
    # Make results reproducible
    np.random.seed(CFG.seed)
    tf.random.set_seed(CFG.seed)
    main()
