from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliceFilterDecision:
    """Explain one slice-selection decision without changing experiment files.

    The paper describes a 55% threshold but its highest/lowest formula is not
    compatible with a percentage threshold.  This project therefore records the
    chosen interpretation explicitly: the largest foreground class share among
    vertebrae, canal, and IVD pixels.  Keeping the calculation in this pure
    function prevents the audit report and the training pipeline from silently
    using different definitions.
    """

    keep: bool
    reason: str
    unique_class_count: int
    dominant_foreground_fraction: float


def foreground_class_fractions(mask: np.ndarray) -> dict[int, float]:
    foreground = mask[mask > 0]
    if foreground.size == 0:
        return {}
    unique, counts = np.unique(foreground, return_counts=True)
    total = counts.sum()
    return {int(label): float(count / total) for label, count in zip(unique, counts)}


def dominant_foreground_fraction(mask: np.ndarray) -> float:
    fractions = foreground_class_fractions(mask)
    # A slice with no foreground must not pass a permissive threshold by
    # accident.  Returning 1.0 preserves the previous conservative behavior.
    return max(fractions.values()) if fractions else 1.0


def evaluate_slice_filter(
    mask: np.ndarray,
    min_classes: int,
    imbalance_threshold: float,
) -> SliceFilterDecision:
    """Apply the documented class-count gate before the imbalance gate."""

    if not 0.0 <= imbalance_threshold <= 1.0:
        raise ValueError("imbalance_threshold must be within [0, 1]")

    unique_class_count = int(np.unique(mask).size)
    dominant_fraction = dominant_foreground_fraction(mask)
    if unique_class_count < min_classes:
        return SliceFilterDecision(
            keep=False,
            reason="fewer_than_min_classes",
            unique_class_count=unique_class_count,
            dominant_foreground_fraction=dominant_fraction,
        )

    # The paper says images above 55% are removed.  Equality is intentionally
    # retained, which makes the boundary testable and avoids an undocumented
    # >= interpretation.
    if dominant_fraction > imbalance_threshold:
        return SliceFilterDecision(
            keep=False,
            reason="dominant_foreground_above_threshold",
            unique_class_count=unique_class_count,
            dominant_foreground_fraction=dominant_fraction,
        )

    return SliceFilterDecision(
        keep=True,
        reason="kept",
        unique_class_count=unique_class_count,
        dominant_foreground_fraction=dominant_fraction,
    )
