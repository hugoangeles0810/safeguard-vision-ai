"""
Evaluation metrics for fall detection.

Includes precision, recall, F1-score, AUC-ROC, and specificity.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_probs: Predicted probabilities (N,), optional for AUC

    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn

    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC-ROC if probabilities provided
    if y_probs is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_probs)

    return metrics


def print_metrics(metrics, prefix=''):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix string (e.g., 'Train', 'Val', 'Test')
    """
    print(f"\n{prefix} Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"    FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")


def check_metric_thresholds(metrics):
    """
    Check if metrics meet target thresholds from project plan.

    Target thresholds:
    - Recall > 0.90 (critical - don't miss falls)
    - Precision > 0.85
    - F1-Score > 0.85
    - AUC-ROC > 0.90
    - Specificity > 0.85

    Args:
        metrics: Dictionary of computed metrics

    Returns:
        passed: Boolean indicating if all thresholds met
        report: String report of threshold checks
    """
    thresholds = {
        'recall': 0.90,
        'precision': 0.85,
        'f1_score': 0.85,
        'specificity': 0.85
    }

    if 'auc_roc' in metrics:
        thresholds['auc_roc'] = 0.90

    passed = True
    report = "Metric Threshold Check:\n"

    for metric_name, threshold in thresholds.items():
        value = metrics.get(metric_name, 0.0)
        status = "PASS" if value >= threshold else "FAIL"
        if value < threshold:
            passed = False
        report += f"  {metric_name}: {value:.4f} >= {threshold:.2f} [{status}]\n"

    return passed, report
