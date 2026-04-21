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

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             classification_report, recall_score)
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
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[available]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
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
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt


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
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    # 1. Sort by ascending predicted probability
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    # 2. Split indices into n_bins equal-count bins
    bins = np.array_split(np.arange(n), n_bins)

    # 3. Weighted sum of |mean_pred - frac_positive|
    ece = 0.0
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        mean_pred = y_prob_sorted[bin_indices].mean()
        frac_pos  = y_true_sorted[bin_indices].mean()
        weight    = len(bin_indices) / n
        ece      += weight * abs(mean_pred - frac_pos)

    return ece


def compare_dt_calibration(X_train, X_test, y_train, y_test):
    """Compare calibration of an unbounded DT vs a depth-5 DT.

    Teaches that pure-leaf trees (unbounded depth) produce extreme
    probabilities -> poor calibration; depth-constrained trees smooth
    probabilities -> better calibration.

    Returns:
        Dict with keys 'ece_unbounded' and 'ece_depth_5' (floats in [0, 1]).
    """
    # Unbounded tree
    dt_unbounded = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt_unbounded.fit(X_train, y_train)
    prob_unbounded = dt_unbounded.predict_proba(X_test)[:, 1]
    ece_unbounded  = compute_ece(np.asarray(y_test), prob_unbounded)

    # Depth-5 tree
    dt_depth5 = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_depth5.fit(X_train, y_train)
    prob_depth5 = dt_depth5.predict_proba(X_test)[:, 1]
    ece_depth5  = compute_ece(np.asarray(y_test), prob_depth5)

    return {"ece_unbounded": ece_unbounded, "ece_depth_5": ece_depth5}


def build_random_forest(X_train, y_train, n_estimators=100, max_depth=10,
                        class_weight=None, random_state=42):
    """Train a RandomForestClassifier.

    Args:
        class_weight: None for default, 'balanced' to reweight the loss
            so minority-class samples count more during training.
        random_state: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    paired = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    return dict(paired)


def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold.

    Standard .predict() uses threshold 0.5. Passing a different threshold
    lets you observe how recall responds to operating-point choice -- which
    is what `class_weight='balanced'` effectively shifts.

    Returns:
        Recall as a float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return recall_score(y_test, y_pred, zero_division=0)


def compute_pr_auc(model, X_test, y_test):
    """PR-AUC (average precision) for the positive class.

    Threshold-independent: measures the model's ability to rank positives
    above negatives across all thresholds. Unlike recall at a specific
    threshold, PR-AUC does not change when you apply class_weight='balanced'
    in a way that merely shifts predicted probabilities uniformly -- the
    ranking is what matters.

    Returns:
        Float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_prob)


def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models on the same axes and save as PNG.

    Args:
        output_path: Destination path (e.g., 'results/pr_curves.png').
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_estimator(
        rf_default, X_test, y_test, ax=ax, name="RF default"
    )
    PrecisionRecallDisplay.from_estimator(
        rf_balanced, X_test, y_test, ax=ax, name="RF balanced"
    )
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    CalibrationDisplay.from_estimator(
        rf_default, X_test, y_test, n_bins=10, ax=ax, name="RF default"
    )
    CalibrationDisplay.from_estimator(
        rf_balanced, X_test, y_test, n_bins=10, ax=ax, name="RF balanced"
    )
    ax.set_title("Calibration Curves")
    ax.legend(loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features.

    Linear models need their inputs on a common scale, otherwise features
    with larger numeric ranges (total_charges ~ 0-9000) swamp features with
    smaller ranges (binary indicators at 0/1). Apply StandardScaler to the
    training features BEFORE calling this function.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train_scaled, y_train)
    return lr


def threshold_sweep(model, X_test, y_test, output_path,
                    thresholds=None):
    """Sweep decision thresholds and plot precision, recall, and F1.

    For each threshold in `thresholds`, computes precision, recall, and F1
    for the positive class using the model's predict_proba scores.
    Saves a single figure to output_path.

    Args:
        model: Fitted classifier with predict_proba.
        X_test: Test features.
        y_test: True binary labels.
        output_path: Where to save the PNG (e.g. 'results/threshold_sweep.png').
        thresholds: Array-like of thresholds to evaluate. Defaults to
                    np.arange(0.1, 0.95, 0.05).

    Returns:
        Dict with keys:
          - best_f1_threshold (float)
          - best_f1 (float)
          - recall80_threshold (float): lowest threshold achieving recall >= 0.80
    """
    from sklearn.metrics import precision_score, f1_score

    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = np.asarray(y_test)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    precisions = np.array(precisions)
    recalls    = np.array(recalls)
    f1s        = np.array(f1s)

    # --- identify key thresholds ---
    best_f1_idx       = int(np.argmax(f1s))
    best_f1_threshold = float(thresholds[best_f1_idx])
    best_f1           = float(f1s[best_f1_idx])

    # Lowest threshold where recall >= 0.80
    recall80_mask = recalls >= 0.80
    recall80_threshold = (
        float(thresholds[recall80_mask][0]) if recall80_mask.any() else None
    )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, marker="o", label="Precision", color="#2196F3")
    ax.plot(thresholds, recalls,    marker="s", label="Recall",    color="#FF9800")
    ax.plot(thresholds, f1s,        marker="^", label="F1",        color="#4CAF50")

    # Mark best F1
    ax.axvline(best_f1_threshold, color="#4CAF50", linestyle="--", linewidth=1.2,
               label=f"Best F1 threshold = {best_f1_threshold:.2f} (F1={best_f1:.3f})")

    # Mark recall >= 0.80 threshold
    if recall80_threshold is not None:
        ax.axvline(recall80_threshold, color="#FF9800", linestyle=":", linewidth=1.2,
                   label=f"Recall \u2265 0.80 threshold = {recall80_threshold:.2f}")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep \u2014 Balanced RF (Precision / Recall / F1)")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- annotation note ---
    note = (
        "Petra Telecom capacity: 200 contacts/month.\n"
        f"Recommended threshold: {best_f1_threshold:.2f} (maximises F1).\n"
        "Lower threshold \u2192 more false positives (wasted calls).\n"
        "Higher threshold \u2192 more false negatives (lost customers)."
    )
    ax.text(0.98, 0.35, note, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "best_f1_threshold":   best_f1_threshold,
        "best_f1":             best_f1,
        "recall80_threshold":  recall80_threshold,
    }


# ===========================================================================
# Tier 2 — Permutation importance vs MDI
# ===========================================================================

def plot_permutation_vs_mdi(rf_model, X_test, y_test, feature_names,
                             output_path, n_repeats=10, random_state=42):
    """Compare MDI (mean decrease impurity) vs permutation importance.

    MDI is biased toward high-cardinality features because continuous
    features have more possible split points. Permutation importance is
    model-agnostic and computed on held-out data, making it more reliable.

    Args:
        rf_model: Fitted RandomForestClassifier.
        X_test: Test features (unscaled).
        y_test: True labels.
        feature_names: List of feature name strings.
        output_path: Where to save the PNG.
        n_repeats: Number of permutation rounds.
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys 'mdi' and 'permutation', each a dict of
        {feature_name: importance} sorted descending.
    """
    from sklearn.inspection import permutation_importance

    # --- MDI importances ---
    mdi = dict(zip(feature_names, rf_model.feature_importances_))

    # --- Permutation importances on test set (scoring = PR-AUC) ---
    perm_result = permutation_importance(
        rf_model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=-1,
    )
    perm = dict(zip(feature_names, perm_result.importances_mean))

    # --- Top-10 features by MDI for display ---
    top10 = sorted(mdi, key=mdi.get, reverse=True)[:10]

    mdi_vals  = np.array([mdi[f]  for f in top10])
    perm_vals = np.array([perm[f] for f in top10])   # raw values, no normalisation

    # --- Side-by-side bar chart ---
    x     = np.arange(len(top10))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, mdi_vals,  width,
           label="MDI (Mean Decrease Impurity)",
           color="#4C9BE8", edgecolor="none")
    ax.bar(x + width / 2, perm_vals, width,
           label="Permutation Importance (PR-AUC)",
           color="#E05252", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(top10, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Importance", fontsize=11)
    ax.set_title(
        "Feature Importance: MDI vs Permutation (Top 10 Features)\n"
        "Balanced Random Forest \u2014 Test Set",
        fontsize=12,
        pad=12,
    )
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    mdi_sorted  = dict(sorted(mdi.items(),  key=lambda kv: kv[1], reverse=True))
    perm_sorted = dict(sorted(perm.items(), key=lambda kv: kv[1], reverse=True))
    return {"mdi": mdi_sorted, "permutation": perm_sorted}


# ===========================================================================
# Tier 3 — Custom voting ensemble
# ===========================================================================

class CustomVotingEnsemble:
    """Majority-vote / averaged-proba ensemble of fitted sklearn classifiers.

    Follows sklearn conventions: fit(), predict(), predict_proba().
    Classifiers are assumed already fitted; fit() only stores classes_.

    Args:
        classifiers: List of (name, fitted_estimator) tuples.
    """

    def __init__(self, classifiers):
        self.classifiers = classifiers   # [(name, model), ...]
        self.classes_    = None

    # ------------------------------------------------------------------
    def _aligned_proba(self, estimator, X):
        """Return (n_samples, 2) array with columns = [P(class=0), P(class=1)]."""
        raw     = estimator.predict_proba(X)
        classes = list(estimator.classes_)
        if classes == [0, 1]:
            return raw
        col0 = classes.index(0)
        col1 = classes.index(1)
        return raw[:, [col0, col1]]

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Store classes_ from y (classifiers assumed already fitted)."""
        self.classes_ = np.unique(y)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X_inputs):
        """Average predict_proba across all classifiers.

        Args:
            X_inputs: Single array (same X for all) OR a list/tuple of
                      arrays, one per classifier.
        """
        probas = []
        for i, (_, clf) in enumerate(self.classifiers):
            Xi = X_inputs[i] if isinstance(X_inputs, (list, tuple)) else X_inputs
            probas.append(self._aligned_proba(clf, Xi))
        return np.mean(np.stack(probas, axis=0), axis=0)   # (n_samples, 2)

    # ------------------------------------------------------------------
    def predict(self, X_inputs):
        """Majority vote: each classifier votes based on its top class.

        Ties (even number of classifiers, equal votes) resolve to class 1
        to favour recall on imbalanced problems.
        """
        votes = []
        for i, (_, clf) in enumerate(self.classifiers):
            Xi = X_inputs[i] if isinstance(X_inputs, (list, tuple)) else X_inputs
            p  = self._aligned_proba(clf, Xi)
            votes.append((p[:, 1] >= 0.5).astype(int))
        votes_arr = np.stack(votes, axis=0)   # (n_clf, n_samples)
        return (votes_arr.sum(axis=0) >= len(self.classifiers) / 2).astype(int)


def evaluate_ensemble(ensemble, X_inputs, y_test):
    """Return PR-AUC and classification report for a CustomVotingEnsemble.

    Args:
        ensemble: Fitted CustomVotingEnsemble.
        X_inputs: Single array or list of arrays (one per classifier).
        y_test: True labels.

    Returns:
        Dict with 'pr_auc' (float) and 'report' (str).
    """
    proba  = ensemble.predict_proba(X_inputs)[:, 1]
    preds  = ensemble.predict(X_inputs)
    pr_auc = average_precision_score(y_test, proba)
    report = classification_report(y_test, preds, zero_division=0,
                                   target_names=["No churn", "Churn"])
    return {"pr_auc": pr_auc, "report": report}


def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test_raw,
                                     X_test_scaled, y_test, feature_names,
                                     min_diff=0.15):
    """Find ONE test sample where RF and LR predicted probabilities differ most.

    The tree-vs-linear capability demonstration. The random forest can
    capture feature interactions, non-monotonic relationships, and threshold
    effects that a linear model cannot express with per-feature coefficients.
    Finding a sample where the two models disagree -- and explaining WHY in
    structural terms -- is the lab's evidence that trees have capabilities
    linear models don't, regardless of aggregate PR-AUC.

    Args:
        rf_model: Trained RF (takes raw features).
        lr_model: Trained LR (takes scaled features).
        X_test_raw: Unscaled test features (what RF consumes).
        X_test_scaled: Scaled test features (what LR consumes).
        y_test: True labels for the test set.
        feature_names: List of feature name strings.
        min_diff: Minimum probability difference to count as disagreement.

    Returns:
        Dict with keys:
          - sample_idx (int): test-set row index of the selected sample
          - feature_values (dict): {name: value} for the sample's features
          - rf_proba (float): RF's predicted P(churn=1)
          - lr_proba (float): LR's predicted P(churn=1)
          - prob_diff (float): |rf_proba - lr_proba|
          - true_label (int): 0 or 1
    """
    rf_proba = rf_model.predict_proba(X_test_raw)[:, 1]
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    diff     = np.abs(rf_proba - lr_proba)

    max_idx = int(np.argmax(diff))
    if diff[max_idx] < min_diff:
        return None

    X_arr = (
        X_test_raw.values if hasattr(X_test_raw, "values") else np.asarray(X_test_raw)
    )
    y_arr = (
        y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    )

    feature_values = {
        name: float(X_arr[max_idx, i]) for i, name in enumerate(feature_names)
    }

    return {
        "sample_idx":     max_idx,
        "feature_values": feature_values,
        "rf_proba":       float(rf_proba[max_idx]),
        "lr_proba":       float(lr_proba[max_idx]),
        "prob_diff":      float(diff[max_idx]),
        "true_label":     int(y_arr[max_idx]),
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
        available = [c for c in NUMERIC_FEATURES if c in X_train.columns]
        plt.figure(figsize=(14, 8))
        plot_tree(dt, feature_names=available, max_depth=3,
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
        available = [c for c in NUMERIC_FEATURES if c in X_train.columns]
        imp = get_feature_importances(rf, available)
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

        # Tier 1: Threshold sweep on balanced RF
        sweep = threshold_sweep(rf_bal, X_test, y_test, "results/threshold_sweep.png")
        print(f"\n--- Tier 1: Threshold Sweep (balanced RF) ---")
        print(f"  Best F1 threshold      : {sweep['best_f1_threshold']:.2f}  (F1={sweep['best_f1']:.3f})")
        print(f"  Recall \u2265 0.80 threshold: {sweep['recall80_threshold']}")
        print(f"  Saved: results/threshold_sweep.png")
        print(
            f"\n  Recommendation (200 contacts/month):\n"
            f"  Use threshold={sweep['best_f1_threshold']:.2f} to maximise F1 \u2014 it balances\n"
            f"  the cost of false positives (wasted retention calls) against\n"
            f"  the opportunity cost of false negatives (churners not contacted).\n"
            f"  If churn prevention revenue >> call cost, lower to "
            f"{sweep['recall80_threshold']} to capture \u226580% of churners,\n"
            f"  accepting more wasted contacts within the 200-call budget."
        )

    # Task 6: Tree-vs-linear disagreement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    lr = build_logistic_regression(X_train_scaled, y_train)
    if rf is not None and lr is not None:
        available = [c for c in NUMERIC_FEATURES if c in X_train.columns]
        d = find_tree_vs_linear_disagreement(
            rf, lr, X_test, X_test_scaled, y_test, available
        )
        if d:
            print(f"\n--- Tree-vs-linear disagreement (sample idx={d['sample_idx']}) ---")
            print(f"  RF P(churn=1)={d['rf_proba']:.3f}  LR P(churn=1)={d['lr_proba']:.3f}")
            print(f"  |diff| = {d['prob_diff']:.3f}   true label = {d['true_label']}")
            print(f"  Feature values: {d['feature_values']}")

    # ------------------------------------------------------------------
    # Tier 2: Permutation importance vs MDI
    # ------------------------------------------------------------------
    if rf_bal is not None:
        print(f"\n--- Tier 2: Permutation Importance vs MDI ---")
        available = [c for c in NUMERIC_FEATURES if c in X_train.columns]
        imp_results = plot_permutation_vs_mdi(
            rf_bal, X_test, y_test, available,
            output_path="results/permutation_vs_mdi.png",
        )
        print("  MDI ranking (top 5):")
        for i, (feat, val) in enumerate(list(imp_results["mdi"].items())[:5]):
            print(f"    {i+1}. {feat:<22s} MDI={val:.4f}")
        print("  Permutation ranking (top 5):")
        for i, (feat, val) in enumerate(list(imp_results["permutation"].items())[:5]):
            print(f"    {i+1}. {feat:<22s} perm={val:.4f}")
        print("  Saved: results/permutation_vs_mdi.png")
        print(
            "\n  When/why they disagree:\n"
            "  MDI counts how often a feature is used for splits weighted by\n"
            "  impurity reduction \u2014 it inflates continuous high-cardinality\n"
            "  features (e.g. total_charges, tenure) because they offer more\n"
            "  candidate split points. Permutation importance shuffles one\n"
            "  feature at a time on held-out data and measures the drop in\n"
            "  PR-AUC, so it reflects true out-of-sample predictive value.\n"
            "  When MDI and permutation agree, the feature is genuinely useful.\n"
            "  When MDI is high but permutation is low, the tree exploited\n"
            "  cardinality noise that doesn't generalise. When permutation is\n"
            "  high but MDI is low, the feature interacts with others in ways\n"
            "  the forest's greedy splits undervalue."
        )

    # ------------------------------------------------------------------
    # Tier 3: Custom voting ensemble
    # ------------------------------------------------------------------
    if rf_bal is not None and lr is not None:
        print(f"\n--- Tier 3: Custom Voting Ensemble ---")
        available = [c for c in NUMERIC_FEATURES if c in X_train.columns]

        # Balanced DT for the ensemble
        dt_bal = build_decision_tree(X_train, y_train,
                                     max_depth=5, random_state=42)

        # Build ensemble: LR needs scaled X, tree models need raw X
        # Pass inputs as a list in the same order as classifiers
        ensemble = CustomVotingEnsemble(classifiers=[
            ("lr",     lr),
            ("dt_bal", dt_bal),
            ("rf_bal", rf_bal),
        ])
        ensemble.fit(X_train, y_train)

        # X_inputs list: one array per classifier
        X_inputs_test  = [X_test_scaled, X_test, X_test]

        ens_eval = evaluate_ensemble(ensemble, X_inputs_test, y_test)
        print(f"  Ensemble PR-AUC : {ens_eval['pr_auc']:.4f}")
        print(f"  Classification report:\n{ens_eval['report']}")

        # Individual model PR-AUCs for comparison
        lr_pr_auc  = average_precision_score(
            y_test, lr.predict_proba(X_test_scaled)[:, 1])
        dt_pr_auc  = average_precision_score(
            y_test, dt_bal.predict_proba(X_test)[:, 1])
        rf_pr_auc  = average_precision_score(
            y_test, rf_bal.predict_proba(X_test)[:, 1])

        print(f"\n  Individual PR-AUC comparison:")
        print(f"    LR          : {lr_pr_auc:.4f}")
        print(f"    DT balanced : {dt_pr_auc:.4f}")
        print(f"    RF balanced : {rf_pr_auc:.4f}")
        print(f"    Ensemble    : {ens_eval['pr_auc']:.4f}  "
              f"({'better' if ens_eval['pr_auc'] > max(lr_pr_auc, dt_pr_auc, rf_pr_auc) else 'not better'} than best individual)")
        print(
            "\n  When to expect ensemble gains:\n"
            "  Ensembling improves when models make uncorrelated errors \u2014\n"
            "  the LR is a linear boundary while the RF captures interactions,\n"
            "  so they can complement each other on different regions of the\n"
            "  feature space. If one model dominates (RF here), the ensemble\n"
            "  may not outperform it because the weaker members add noise.\n"
            "  Gains are most reliable when: (a) individual models have\n"
            "  similar PR-AUC, and (b) their errors don't coincide."
        )


if __name__ == "__main__":
    main()