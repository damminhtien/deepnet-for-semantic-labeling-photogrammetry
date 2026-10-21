"""Historical pixel metrics used by pre-pipeline experiments.

New scene-level evaluation and provenance belong in ``deepnet.data.evaluation``.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

from .constants import IGNORE_LABEL, LABELS


def _valid_pairs(predictions, ground_truth, label_values):
    predictions = np.asarray(predictions).ravel()
    ground_truth = np.asarray(ground_truth).ravel()
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            "Predictions and ground truth must have the same number of pixels; "
            f"got {predictions.size} and {ground_truth.size}"
        )

    valid = ground_truth != IGNORE_LABEL
    predictions = predictions[valid]
    ground_truth = ground_truth[valid]
    if ground_truth.size == 0:
        raise ValueError("No non-ignored ground-truth pixels are available")

    expected = np.arange(len(label_values))
    if not np.isin(ground_truth, expected).all():
        raise ValueError("Ground truth contains labels outside the configured classes")
    if not np.isin(predictions, expected).all():
        raise ValueError("Predictions contain labels outside the configured classes")
    return predictions, ground_truth


def accuracy(predictions, ground_truth):
    """Return overall accuracy as a percentage, excluding ignored pixels."""
    predictions, ground_truth = _valid_pairs(predictions, ground_truth, LABELS)
    return float(np.mean(predictions == ground_truth) * 100.0)


def metrics(predictions, ground_truth, label_values=LABELS):
    """Print the historical metric report and return overall accuracy."""
    predictions, ground_truth = _valid_pairs(predictions, ground_truth, label_values)
    labels = np.arange(len(label_values))
    matrix = confusion_matrix(ground_truth, predictions, labels=labels)

    print("Confusion matrix:")
    print(matrix)
    print("---")

    total = int(matrix.sum())
    score = float(np.trace(matrix) * 100.0 / total)
    print(f"{total} pixels processed")
    print(f"Total accuracy: {score}%")
    print("---")

    denominators = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1_scores = np.divide(
        2.0 * np.diag(matrix),
        denominators,
        out=np.zeros(len(label_values), dtype=float),
        where=denominators != 0,
    )
    print("F1 score:")
    for label, f1_score in zip(label_values, f1_scores, strict=False):
        print(f"{label}: {f1_score}")
    print("---")

    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    observed_agreement = np.trace(matrix) / total
    expected_agreement = float(np.dot(actual, predicted)) / (total * total)
    kappa_denominator = 1.0 - expected_agreement
    kappa = (
        (observed_agreement - expected_agreement) / kappa_denominator
        if kappa_denominator
        else float("nan")
    )
    print(f"Kappa: {kappa}")
    return score
