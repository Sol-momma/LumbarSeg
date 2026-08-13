"""Auditable reconstruction of the authors' public APTA mask operations.

This module is deliberately isolated from the default preprocessing path.  The
public author code applies linear interpolation and repeated neighborhood voting
to segmentation masks, so adopting it without a side-by-side audit could erase
small spinal-canal regions.  Functions here reproduce those observable steps for
comparison; they do not claim to fill the unpublished parts of the paper method.
"""

from __future__ import annotations

import numpy as np


APTA_AUDIT_DEFINITION = "public_author_code_reconstruction"

# PIL iterates x first and y second in the source implementation.  The ordering
# matters when two colors have the same count because Counter.most_common keeps
# the first color encountered.  NumPy indexes rows (y) before columns (x), hence
# the explicit (dy, dx) conversion below.
FOUR_NEIGHBOR_OFFSETS = ((0, 1), (0, -1), (1, 0), (-1, 0))
EIGHT_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def normalize_mask_like_author(mask: np.ndarray) -> np.ndarray:
    """Apply the public code's per-slice 0..255 normalization safely."""

    if mask.ndim != 2:
        raise ValueError(f"APTA expects a 2D slice; got shape {mask.shape}")
    minimum = float(np.min(mask))
    maximum = float(np.max(mask))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("APTA input contains a non-finite value")
    if maximum == minimum:
        # The public script divides by zero and silently casts NaN to uint8.
        # Audits reject that undefined result so it cannot look reproducible.
        raise ValueError("APTA cannot normalize a constant slice")
    normalized = (mask.astype(np.float32) - minimum) / (maximum - minimum) * 255.0
    return normalized.astype(np.uint8)


def threshold_author_colors(grayscale: np.ndarray) -> np.ndarray:
    """Map author grayscale ranges to black/red/green/blue labels 0..3."""

    if grayscale.ndim != 2:
        raise ValueError(f"APTA thresholding expects 2D input; got shape {grayscale.shape}")
    labels = np.zeros(grayscale.shape, dtype=np.uint8)
    labels[(grayscale >= 1) & (grayscale <= 10)] = 1
    labels[(grayscale >= 90) & (grayscale <= 180)] = 2
    # Although the source says 180..255 for blue, its preceding ``elif`` makes
    # 180 green.  Starting at 181 preserves the actual executed behavior.
    labels[(grayscale >= 181) & (grayscale <= 255)] = 3
    return labels


def _neighbor_stack(labels: np.ndarray, offsets: tuple[tuple[int, int], ...]) -> np.ndarray:
    if labels.ndim != 2 or min(labels.shape) < 2:
        raise ValueError(f"Neighborhood operations require at least a 2x2 image; got {labels.shape}")
    height, width = labels.shape
    stack = np.full((len(offsets), height, width), -1, dtype=np.int16)
    for rank, (dy, dx) in enumerate(offsets):
        destination_y = slice(max(0, -dy), min(height, height - dy))
        destination_x = slice(max(0, -dx), min(width, width - dx))
        source_y = slice(max(0, dy), min(height, height + dy))
        source_x = slice(max(0, dx), min(width, width + dx))
        stack[rank, destination_y, destination_x] = labels[source_y, source_x]
    return stack


def _ordered_neighbor_mode(
    labels: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = _neighbor_stack(labels, offsets)
    best_label = np.zeros(labels.shape, dtype=np.uint8)
    best_count = np.full(labels.shape, -1, dtype=np.int16)
    best_first_rank = np.full(labels.shape, len(offsets) + 1, dtype=np.int16)
    distinct_count = np.zeros(labels.shape, dtype=np.uint8)

    for label in range(4):
        matches = neighbors == label
        count = matches.sum(axis=0, dtype=np.int16)
        distinct_count += count > 0
        ranks = np.where(matches, np.arange(len(offsets))[:, None, None], len(offsets) + 1)
        first_rank = ranks.min(axis=0)
        replace = (count > best_count) | ((count == best_count) & (first_rank < best_first_rank))
        best_label[replace] = label
        best_count[replace] = count[replace]
        best_first_rank[replace] = first_rank[replace]

    return best_label, best_count, distinct_count


def apply_public_apta_neighborhoods(labels: np.ndarray) -> np.ndarray:
    """Apply the five public neighborhood/color consolidation stages."""

    if labels.dtype.kind not in "ui" or labels.size == 0 or not np.isin(labels, np.arange(4)).all():
        raise ValueError("APTA labels must be a non-empty integer array containing only 0..3")

    # Stage 1: remove a pixel when none of its eight neighbors has the same color.
    neighbors = _neighbor_stack(labels, EIGHT_NEIGHBOR_OFFSETS)
    has_same_neighbor = (neighbors == labels[None, ...]).any(axis=0)
    current = np.where(has_same_neighbor, labels, 0).astype(np.uint8)

    # Stage 2: the function named remove_outline replaces every pixel with the
    # four-neighbor mode.  This is stronger than an edge-only operation.
    current = _ordered_neighbor_mode(current, FOUR_NEIGHBOR_OFFSETS)[0]

    # Stage 3: replace a differing boundary pixel only when at least two colors
    # occur in its eight-neighbor set.
    mode, _, distinct_count = _ordered_neighbor_mode(current, EIGHT_NEIGHBOR_OFFSETS)
    current = np.where((distinct_count >= 2) & (current != mode), mode, current).astype(np.uint8)

    # Stage 4: replace any remaining singleton with the ordered eight-neighbor mode.
    neighbors = _neighbor_stack(current, EIGHT_NEIGHBOR_OFFSETS)
    has_same_neighbor = (neighbors == current[None, ...]).any(axis=0)
    mode = _ordered_neighbor_mode(current, EIGHT_NEIGHBOR_OFFSETS)[0]
    current = np.where(has_same_neighbor, current, mode).astype(np.uint8)

    # Stage 5: the public code collapses green and blue to red when red has
    # disappeared entirely from the slice.
    if not np.any(current == 1) and np.any((current == 2) | (current == 3)):
        current = current.copy()
        current[(current == 2) | (current == 3)] = 1
    return current


def reconstruct_public_apta(
    raw_mask_slice: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    apply_author_flip: bool,
) -> np.ndarray:
    """Reconstruct the public normalization, resize, thresholds, and voting."""

    # Keep OpenCV optional at module import time. Pure threshold/neighborhood
    # tests run on the Mac review environment, while pixel-identical resizing is
    # exercised in the WSL training environment where OpenCV is installed.
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the full APTA reconstruction") from exc
    normalized = normalize_mask_like_author(raw_mask_slice)
    if apply_author_flip:
        # ROTATE_180 followed by horizontal flip equals an up/down flip.  Keeping
        # this optional separates author-geometry diagnosis from fair comparisons
        # that use this project's verified coordinates for both branches.
        normalized = np.flipud(normalized)
    resized = cv2.resize(
        normalized,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return apply_public_apta_neighborhoods(threshold_author_colors(resized))
