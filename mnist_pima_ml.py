#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mnist_pima_ml.py
---------------------------------
WA3 — Two problems in one script (clean & reproducible)

Problem 1 — MNIST Classification & Clustering
  • Load MNIST (OpenML), stratified split (train=10k, test=2k)
  • Decision Tree with GridSearchCV (max_features, max_depth)
  • PCA (retain 90% variance) + Decision Tree
  • Gradient Boosting (depth=2, n_estimators=6, learning_rate=1.0) on PCA data
  • PCA reconstruction (first 5 test digits) vs originals
  • KMeans (k=20) on PCA features; visualize centroids
  • Cluster-based labeling (majority label per cluster) → accuracy & F1

Problem 2 — Pima Indians Diabetes Classification
  • Fetch dataset (OpenML), treat zeros as missing (select columns), fill with median
  • Stratified split: 700 train / rest test
  • Train/evaluate: Decision Tree, Random Forest, Bagging(SVM, 10 estimators), AdaBoost(DT base, 100, lr=0.25)
  • Metrics: Accuracy, Precision, Recall, F1; Confusion matrices
  • Semi-supervised: SelfTrainingClassifier(RandomForest) with 200 labeled; compare vs supervised

Outputs
  • Figures → ./figures
  • Reports (JSON) → ./reports

Run
  python mnist_pima_ml.py
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml

FIGURES_DIR = "figures"
REPORTS_DIR = "reports"
RNG_SEED = 42


# ---------- utils ----------
def safe_makedirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, fname: str) -> str:
    plt.figure(figsize=(4.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")
    plt.title(title)
    out = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ==================================================
# Problem 1 — MNIST Classification & Clustering
# ==================================================
def problem1_mnist(seed: int = RNG_SEED) -> Dict:
    # Load MNIST from OpenML
    print("[MNIST] Fetching MNIST (this may take a moment on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(int)

    # Standardize pixel values to [0,1]
    X /= 255.0

    # Create a smaller stratified subset as per assignment (10k train, 2k test)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=10_000, stratify=y, random_state=seed
    )
    X_test, _, y_test, _ = train_test_split(
        X_rest, y_rest, test_size=2_000, stratify=y_rest, random_state=seed
    )

    # ---- Decision Tree with GridSearch ----
    dt = DecisionTreeClassifier(random_state=seed)
    param_grid = {
        "max_features": [100, 150, 200],
        "max_depth": [2, 4, 5],
    }
    grid = GridSearchCV(dt, param_grid, cv=3, scoring="accuracy", n_jobs=None)
    grid.fit(X_train, y_train)
    dt_best = grid.best_estimator_
    y_pred_dt = dt_best.predict(X_test)

    dt_metrics = {
        "best_params": grid.best_params_,
        "accuracy": float(accuracy_score(y_test, y_pred_dt)),
        "f1_macro": float(f1_score(y_test, y_pred_dt, average="macro")),
    }

    # ---- PCA (retain 90% variance) + Decision Tree ----
    pca = PCA(n_components=0.90, svd_solver="full", random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    dt_pca = GridSearchCV(DecisionTreeClassifier(random_state=seed), param_grid, cv=3, scoring="accuracy")
    dt_pca.fit(X_train_pca, y_train)
    y_pred_dt_pca = dt_pca.best_estimator_.predict(X_test_pca)

    dt_pca_metrics = {
        "n_components": int(pca.n_components_),
        "best_params": dt_pca.best_params_,
        "accuracy": float(accuracy_score(y_test, y_pred_dt_pca)),
        "f1_macro": float(f1_score(y_test, y_pred_dt_pca, average="macro")),
    }

    # ---- Gradient Boosting on PCA features ----
    gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=6, learning_rate=1.0, random_state=seed)
    gbrt.fit(X_train_pca, y_train)
    y_pred_gbrt = gbrt.predict(X_test_pca)
    gbrt_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_gbrt)),
        "f1_macro": float(f1_score(y_test, y_pred_gbrt, average="macro")),
    }

    # ---- PCA reconstruction (first 5 test digits) ----
    n_show = 5
    Xr = pca.inverse_transform(X_test_pca[:n_show])
    # plot side-by-side originals vs reconstructions
    fig, axes = plt.subplots(2, n_show, figsize=(2*n_show, 4))
    for i in range(n_show):
        axes[0, i].imshow(X_test[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Orig {y_test[i]}")
        axes[1, i].imshow(Xr[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Recon")
    plt.tight_layout()
    recon_path = os.path.join(FIGURES_DIR, "mnist_pca_recon_first5.png")
    plt.savefig(recon_path, dpi=150)
    plt.close()

    # ---- KMeans on PCA features ----
    k = 20
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    kmeans.fit(X_train_pca)
    # visualize centroid images
    centroids_img = pca.inverse_transform(kmeans.cluster_centers_).reshape(k, 28, 28)
    grid_cols = 5
    grid_rows = math.ceil(k / grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(2*grid_cols, 2*grid_rows))
    axes = axes.ravel()
    for i in range(k):
        axes[i].imshow(centroids_img[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"C{i}")
    for j in range(k, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    centroids_path = os.path.join(FIGURES_DIR, "mnist_kmeans_centroids.png")
    plt.savefig(centroids_path, dpi=150)
    plt.close()

    # ---- Cluster-based labeling (majority vote per cluster) ----
    train_clusters = kmeans.predict(X_train_pca)
    test_clusters = kmeans.predict(X_test_pca)
    cluster_to_label = {}
    for c in range(k):
        mask = (train_clusters == c)
        if mask.sum() == 0:
            cluster_to_label[c] = 0
        else:
            # majority label
            vals, counts = np.unique(y_train[mask], return_counts=True)
            cluster_to_label[c] = int(vals[np.argmax(counts)])

    y_pred_cluster = np.array([cluster_to_label[c] for c in test_clusters])
    cl_acc = float(accuracy_score(y_test, y_pred_cluster))
    cl_f1 = float(f1_score(y_test, y_pred_cluster, average="macro"))

    # collect report
    report = {
        "decision_tree": dt_metrics,
        "pca_dt": dt_pca_metrics,
        "gbrt_on_pca": gbrt_metrics,
        "clustering": {"k": k, "accuracy": cl_acc, "f1_macro": cl_f1},
        "figures": {"pca_recon_first5": recon_path, "kmeans_centroids": centroids_path},
        "pca_components": int(pca.n_components_),
    }
    return report


# ==================================================
# Problem 2 — Pima Indians Diabetes Classification
# ==================================================
def fetch_pima_openml() -> Tuple[pd.DataFrame, pd.Series]:
    # Pima Indians Diabetes on OpenML (768 rows, binary class)
    # Multiple versions exist; try common names/ids.
    # Prefer "diabetes" with 768 samples; ensure we grab classification target.
    ds = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
    X = ds.frame.copy()
    # Try to locate target column
    if "class" in X.columns:
        y = X.pop("class")
    elif "Class" in X.columns:
        y = X.pop("Class")
    elif "Outcome" in X.columns:
        y = X.pop("Outcome")
    else:
        # fallback: last column
        y = X.iloc[:, -1]
        X = X.iloc[:, :-1]
    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    return X, y


def clean_pima_zeros(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    # Known columns where zeros are invalid in Pima
    candidates = ["plas", "pres", "skin", "insu", "mass",  # openml naming variant
                  "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]  # kaggle naming
    for col in candidates:
        if col in Xc.columns:
            non_zero = Xc[col] != 0
            if non_zero.any():
                med = Xc.loc[non_zero, col].median()
                Xc.loc[~non_zero, col] = med
    return Xc


def evaluate_classifier(model, X_tr, y_tr, X_te, y_te, title_prefix: str) -> Dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    cm = pd.crosstab(pd.Series(y_te, name="True"), pd.Series(y_pred, name="Pred"))
    cm_path = plot_confusion_matrix(cm.values, labels=sorted(np.unique(y_te)), title=f"{title_prefix} — Confusion Matrix", fname=f"pima_cm_{title_prefix.replace(' ', '_').lower()}.png")
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred)),
        "recall": float(recall_score(y_te, y_pred)),
        "f1": float(f1_score(y_te, y_pred)),
        "confusion_matrix_path": cm_path,
    }


def problem2_pima(seed: int = RNG_SEED) -> Dict:
    X, y = fetch_pima_openml()
    # Drop rows with missing (if any)
    df = pd.concat([X, y.rename("target")], axis=1).dropna()
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Clean invalid zeros
    X = clean_pima_zeros(X)

    # Stratified split: 700 train / rest test
    test_size = max(0, len(X) - 700) / len(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Standardization for some models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=seed)
    results["DecisionTree"] = evaluate_classifier(dt, X_train, y_train, X_test, y_test, "Decision Tree")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    results["RandomForest"] = evaluate_classifier(rf, X_train, y_train, X_test, y_test, "Random Forest")

    # Bagging with SVM (linear kernel) — 10 estimators
    base_svm = SVC(kernel="linear", probability=False, random_state=seed)
    bag_svm = BaggingClassifier(base_estimator=base_svm, n_estimators=10, random_state=seed, n_jobs=None)
    results["Bagging_SVM"] = evaluate_classifier(bag_svm, X_train_s, y_train, X_test_s, y_test, "Bagging SVM")

    # AdaBoost with Decision Tree base (stumps depth=1) — 100 estimators, lr=0.25
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    ada = AdaBoostClassifier(base_estimator=stump, n_estimators=100, learning_rate=0.25, random_state=seed)
    results["AdaBoost_DT"] = evaluate_classifier(ada, X_train, y_train, X_test, y_test, "AdaBoost DT")

    # Semi-supervised: SelfTraining with RF, 200 labeled samples
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X_train))
    rng.shuffle(idx)
    labeled_idx = idx[:200]
    unlabeled_idx = idx[200:]

    y_semi = y_train.copy().to_numpy()
    y_semi[unlabeled_idx] = -1  # mark unlabeled

    base_rf = RandomForestClassifier(n_estimators=200, random_state=seed)
    self_train = SelfTrainingClassifier(base_rf, threshold=0.99, verbose=False)
    self_train.fit(X_train, y_semi)

    y_pred_semi = self_train.predict(X_test)
    cm_semi = pd.crosstab(pd.Series(y_test, name="True"), pd.Series(y_pred_semi, name="Pred"))
    cm_semi_path = plot_confusion_matrix(cm_semi.values, labels=sorted(np.unique(y_test)), title="Self-Training RF — Confusion Matrix", fname="pima_cm_selftraining.png")
    semi_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_semi)),
        "precision": float(precision_score(y_test, y_pred_semi)),
        "recall": float(recall_score(y_test, y_pred_semi)),
        "f1": float(f1_score(y_test, y_pred_semi)),
        "confusion_matrix_path": cm_semi_path,
        "labeled_samples": int(len(labeled_idx)),
    }

    results["SelfTraining_RF"] = semi_metrics

    return results


# --------------- main ---------------
def main() -> None:
    safe_makedirs(FIGURES_DIR)
    safe_makedirs(REPORTS_DIR)

    mnist_report = problem1_mnist()
    pima_report = problem2_pima()

    out = {"problem1_mnist": mnist_report, "problem2_pima": pima_report}
    save_json(out, os.path.join(REPORTS_DIR, "mnist_pima_report.json"))
    print("Done. Reports saved to reports/mnist_pima_report.json; figures saved under figures/.")


if __name__ == "__main__":
    main()
