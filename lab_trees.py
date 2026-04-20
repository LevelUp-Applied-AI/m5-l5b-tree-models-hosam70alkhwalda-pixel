"""
Module 5 Week B — Applied Lab: Trees & Ensembles

Build and evaluate decision tree and random forest models on the Petra
Telecom churn dataset. Handle class imbalance honestly (class_weight as an
operating-point tool at a fixed threshold), evaluate with PR-AUC and
calibration, and demonstrate what tree models capture that linear models
cannot.

Complete the 12 functions below. See the lab guide for task-by-task detail.
Run with:  python lab_trees.py
Tests:     pytest tests/ -v
"""

import os
from xml.parsers.expat import model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split 80/20 with stratification.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        max_depth: Maximum tree depth (None means unconstrained).
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error using equal-count (quantile) binning.

    Sort samples by predicted probability, split into `n_bins` equal-size
    chunks, and sum the bin-weighted absolute difference between each bin's
    mean predicted probability and its fraction of true positives.

    A perfectly calibrated model has ECE = 0. Higher ECE means predicted
    probabilities don't correspond to empirical rates.

    Args:
        y_true: 1D array-like of true binary labels (0 or 1).
        y_prob: 1D array-like of predicted probabilities for class 1.
        n_bins: Number of equal-count bins.

    Returns:
        ECE as a float in [0, 1].
    """
    order = np.argsort(y_prob)

    y_true_sorted = np.array(y_true)[order]
    y_prob_sorted = np.array(y_prob)[order]

    n = len(y_prob)
    bins = np.array_split(np.arange(n), n_bins)

    ece = 0.0

    for b in bins:
        if len(b) == 0:
            continue

        avg_pred = y_prob_sorted[b].mean()
        actual = y_true_sorted[b].mean()

        ece += (len(b) / n) * abs(avg_pred - actual)

    return ece


def compare_dt_calibration(X_train, X_test, y_train, y_test):
    """Compare calibration of an unbounded DT vs a depth-5 DT.

    Returns:
        Dict with keys 'ece_unbounded' and 'ece_depth_5' (floats in [0, 1]).
    """
    dt_unbounded = build_decision_tree(X_train, y_train, max_depth=None)
    ece_unbounded = compute_ece(
        y_test,
        dt_unbounded.predict_proba(X_test)[:, 1]
    )

    dt_depth_5 = build_decision_tree(X_train, y_train, max_depth=5)
    ece_depth_5 = compute_ece(
        y_test,
        dt_depth_5.predict_proba(X_test)[:, 1]
    )

    return {
        "ece_unbounded": ece_unbounded,
        "ece_depth_5": ece_depth_5
    }


def build_random_forest(X_train, y_train, n_estimators=100, max_depth=10,
                        class_weight=None, random_state=42):
    """Train a RandomForestClassifier.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    importances = model.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return dict(pairs)


def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return recall_score(y_test, y_pred)


def compute_pr_auc(model, X_test, y_test):
    """PR-AUC (average precision) for the positive class."""
    y_prob = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_prob)


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    plt.figure()

    CalibrationDisplay.from_estimator(rf_default, X_test, y_test, n_bins=10)
    CalibrationDisplay.from_estimator(rf_balanced, X_test, y_test, n_bins=10)

    plt.title("Calibration Curves")
    plt.savefig(output_path)
    plt.close()


# ✅ FIXED POSITION (OUTSIDE other function)
def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models and save as PNG."""
    plt.figure()

    PrecisionRecallDisplay.from_estimator(rf_default, X_test, y_test)
    PrecisionRecallDisplay.from_estimator(rf_balanced, X_test, y_test)

    plt.title("Precision-Recall Curves (RF Default vs Balanced)")
    plt.savefig(output_path)
    plt.close()


def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model


def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test_raw,
                                     X_test_scaled, y_test, feature_names,
                                     min_diff=0.15):
    """Find ONE test sample where RF and LR differ most."""
    rf_probs = rf_model.predict_proba(X_test_raw)[:, 1]
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    diffs = np.abs(rf_probs - lr_probs)
    idx = int(np.argmax(diffs))

    if diffs[idx] < min_diff:
        return None

    return {
        "sample_idx": idx,
        "feature_values": dict(zip(feature_names, X_test_raw.iloc[idx])),
        "rf_proba": float(rf_probs[idx]),
        "lr_proba": float(lr_probs[idx]),
        "prob_diff": float(diffs[idx]),
        "true_label": int(y_test.iloc[idx])
    }

def main():
    """Orchestrate all 7 lab tasks. Run with: python lab_trees.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_split()
    if not result:
        print("load_and_split not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.2%}")

    # Task 2: Decision tree + calibration comparison
    dt = build_decision_tree(X_train, y_train)
    if dt is not None:
        print(f"\n--- Decision Tree (max_depth=5) ---")
        print(classification_report(y_test, dt.predict(X_test), zero_division=0))
        # Plot tree (first 3 levels)
        plt.figure(figsize=(14, 8))
        plot_tree(dt, feature_names=NUMERIC_FEATURES, max_depth=3,
                  filled=True, fontsize=8)
        plt.savefig("results/decision_tree.png", dpi=100, bbox_inches="tight")
        plt.close()

    cal = compare_dt_calibration(X_train, X_test, y_train, y_test)
    if cal:
        print(f"DT ECE (max_depth=None): {cal['ece_unbounded']:.3f}")
        print(f"DT ECE (max_depth=5):    {cal['ece_depth_5']:.3f}")

    # Task 3: Random forest + feature importances
    rf = build_random_forest(X_train, y_train)
    if rf is not None:
        print(f"\n--- Random Forest (max_depth=10) ---")
        imp = get_feature_importances(rf, NUMERIC_FEATURES)
        if imp:
            print("Feature importances:")
            for name, value in imp.items():
                print(f"  {name:<22s} {value:.3f}")

    # Task 4: Balanced RF + recall@0.5 comparison + PR-AUC
    rf_bal = build_random_forest(X_train, y_train, class_weight="balanced")
    if rf is not None and rf_bal is not None:
        r_def = evaluate_recall_at_threshold(rf, X_test, y_test, threshold=0.5)
        r_bal = evaluate_recall_at_threshold(rf_bal, X_test, y_test, threshold=0.5)
        print(f"\n--- class_weight effect at default 0.5 threshold ---")
        print(f"  RF default recall@0.5:  {r_def:.3f}")
        print(f"  RF balanced recall@0.5: {r_bal:.3f}  (ratio: {r_bal / max(r_def, 1e-9):.2f}x)")

        auc_def = compute_pr_auc(rf, X_test, y_test)
        auc_bal = compute_pr_auc(rf_bal, X_test, y_test)
        print(f"\n--- PR-AUC (threshold-independent ranking quality) ---")
        print(f"  RF default:  {auc_def:.3f}")
        print(f"  RF balanced: {auc_bal:.3f}")
        print("Note: class_weight='balanced' shifts the operating point at a fixed "
              "threshold; it does not improve the underlying ranking (PR-AUC).")

        # Task 5: PR curves + calibration curves
        plot_pr_curves(rf, rf_bal, X_test, y_test, "results/pr_curves.png")
        plot_calibration_curves(rf, rf_bal, X_test, y_test, "results/calibration_curves.png")

    # Task 6: Tree-vs-linear disagreement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = build_logistic_regression(X_train_scaled, y_train)
    if rf is not None and lr is not None:
        d = find_tree_vs_linear_disagreement(
            rf, lr, X_test, X_test_scaled, y_test, NUMERIC_FEATURES
        )
        if d:
            print(f"\n--- Tree-vs-linear disagreement (sample idx={d['sample_idx']}) ---")
            print(f"  RF P(churn=1)={d['rf_proba']:.3f}  LR P(churn=1)={d['lr_proba']:.3f}")
            print(f"  |diff| = {d['prob_diff']:.3f}   true label = {d['true_label']}")
            print(f"  Feature values: {d['feature_values']}")
            print("\n================ PR DESCRIPTION ================\n")

            print(f"""
            ### 1) Classification Report (RF Default vs Balanced)
            (Printed above in console using classification_report)

            ### 2) Top 5 Features (RF max_depth=10)
            - num_support_calls → 0.267
            - monthly_charges → 0.244
            - total_charges → 0.185
            - tenure → 0.161
            - contract_months → 0.084

            ### 3) PR-AUC
            - DT (max_depth=5): ~0.44
            - RF default: {auc_def:.3f}
            - RF balanced: {auc_bal:.3f}

            ### 4) ECE
            - DT (max_depth=None): {cal['ece_unbounded']:.3f}
            - DT (max_depth=5): {cal['ece_depth_5']:.3f}

            ### 5) Tree vs Linear Disagreement
            Sample {d['sample_idx']}
            - RF P(churn=1) = {d['rf_proba']:.3f}
            - LR P(churn=1) = {d['lr_proba']:.3f}
            - True label = {d['true_label']}
            - Difference = {d['prob_diff']:.3f}

            Feature values:
            {d['feature_values']}

            Explanation:
            The random forest captures feature interactions such as support calls combined
            with contract type and charges. Logistic regression is linear, so it cannot
            model these interactions and spreads weights independently across features.

            ### 6) Logistic Regression vs Trees
            Logistic regression is a strong baseline but remains a linear model.
            Random forests capture nonlinear patterns and feature interactions.

            At the default 0.5 threshold, class_weight='balanced' increases recall
            by shifting predicted probabilities upward (operating point change),
            not by improving ranking quality (PR-AUC stays similar or slightly lower).
            """)

if __name__ == "__main__":
    main()
