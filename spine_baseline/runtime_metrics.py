from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNTIME_METRICS_SCHEMA_VERSION = 1
PROCESSED_REVIEW_SCOPE = "processed_slices_to_review_png"


def build_runtime_report(
    *,
    stage_seconds: Mapping[str, float],
    total_seconds: float,
    input_files: Sequence[str],
    scored_slice_count: int,
    rendered_panel_count: int,
    batch_size: int,
    target_height: int,
    target_width: int,
    split: str,
    model_path: Path,
    processed_root: Path,
    gpu_devices: Sequence[str],
) -> dict[str, Any]:
    """Build an explicit, machine-readable timing contract for review PNGs.

    This report deliberately names its scope as *processed slices* rather than
    end-to-end MRI inference. The current visualization command requires cached
    image/mask pairs, so calling this an MHA-to-result latency would overstate
    what was measured and could lead to an invalid social-deployment claim.
    """
    normalized_stages = {
        name: round(max(0.0, float(seconds)), 6)
        for name, seconds in stage_seconds.items()
    }
    measured_stage_seconds = sum(normalized_stages.values())
    normalized_total = round(max(0.0, float(total_seconds)), 6)
    series_ids = {filename.rsplit("_s", 1)[0] for filename in input_files}

    return {
        "schema_version": RUNTIME_METRICS_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": PROCESSED_REVIEW_SCOPE,
        "scope_includes": [
            "processed slice selection",
            "model loading",
            "processed NPZ loading",
            "model inference",
            "prediction scoring",
            "review PNG rendering and saving",
            "CSV summary writing",
        ],
        "scope_excludes": [
            "raw MHA loading",
            "sagittal orientation resolution",
            "MHA-to-NPZ preprocessing",
            "clinical-system transfer such as PACS or network download",
        ],
        "timing_semantics": {
            "total_seconds": "One command invocation, including model load and first inference overhead.",
            "stage_seconds": "Non-overlapping measured stages inside total_seconds.",
            "unattributed_seconds": "Command overhead not enclosed by a named stage.",
        },
        "total_seconds": normalized_total,
        "stage_seconds": normalized_stages,
        "unattributed_seconds": round(max(0.0, normalized_total - measured_stage_seconds), 6),
        "counts": {
            "input_slices": len(input_files),
            "input_series": len(series_ids),
            "scored_slices": int(scored_slice_count),
            "rendered_panels": int(rendered_panel_count),
        },
        "configuration": {
            "batch_size": int(batch_size),
            "target_height": int(target_height),
            "target_width": int(target_width),
            "split": split,
            "model_path": str(model_path),
            "processed_root": str(processed_root),
            "gpu_devices": list(gpu_devices),
        },
    }


def write_runtime_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write through a sibling temporary file so interruption cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
