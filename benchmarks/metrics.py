"""
Metrics for benchmark runs.

Pure functions. No I/O. No dataset knowledge. Takes lists of
predictions and ground-truths, returns numbers.
"""

from __future__ import annotations

import statistics


def mape(predictions: list[float], truths: list[float]) -> float:
    """
    Mean absolute percentage error, in percent.

    Zero truths are skipped — division by zero has no meaning.
    """
    ratios = [
        abs(p - t) / t
        for p, t in zip(predictions, truths)
        if t != 0
    ]
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios) * 100.0


def median_ape(predictions: list[float], truths: list[float]) -> float:
    """Median absolute percentage error. More robust to outliers than MAPE."""
    ratios = [
        abs(p - t) / t * 100.0
        for p, t in zip(predictions, truths)
        if t != 0
    ]
    if not ratios:
        return 0.0
    return statistics.median(ratios)


def signed_bias(predictions: list[float], truths: list[float]) -> float:
    """
    Mean signed percentage error. Positive means we overestimate.
    """
    ratios = [
        (p - t) / t * 100.0
        for p, t in zip(predictions, truths)
        if t != 0
    ]
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def mae(predictions: list[float], truths: list[float]) -> float:
    """Mean absolute error, in the same unit as the inputs."""
    if not predictions:
        return 0.0
    return sum(abs(p - t) for p, t in zip(predictions, truths)) / len(predictions)


def top1_accuracy(predicted: list[str], truths: list[str]) -> float:
    """Fraction of exact label matches, case-insensitive."""
    if not predicted:
        return 0.0
    correct = sum(
        1 for p, t in zip(predicted, truths) if p.strip().lower() == t.strip().lower()
    )
    return correct / len(predicted)


def summarize(
    predictions: list[float],
    truths: list[float],
    predictions_labels: list[str] | None = None,
    truths_labels: list[str] | None = None,
) -> dict:
    """Return every metric in a single dict."""
    out = {
        "count": len(predictions),
        "mape": round(mape(predictions, truths), 2),
        "median_ape": round(median_ape(predictions, truths), 2),
        "signed_bias": round(signed_bias(predictions, truths), 2),
        "mae": round(mae(predictions, truths), 2),
    }
    if predictions_labels and truths_labels:
        out["top1_accuracy"] = round(
            top1_accuracy(predictions_labels, truths_labels), 4
        )
    return out