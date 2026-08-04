from __future__ import annotations

from argparse import ArgumentParser
from csv import DictReader
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys


# These are the class-wise T2 SPACE Dice values reported in Tables 2 and 3 of
# Ahmed et al. The campaign must preserve the individual floors: otherwise a
# very strong large structure could hide a regression in the smaller canal.
PAPER_DICE_TARGETS = {
    "IVDs": 0.9688,
    "Vertebrae": 0.9712,
    "Spinal Canal": 0.9671,
}
FOREGROUND_MACRO_TARGET = 0.9700


class TargetEvidenceError(ValueError):
    """Raised when a metrics CSV cannot support a reproducible decision."""


@dataclass(frozen=True)
class TargetCheck:
    score_target_met: bool
    foreground_macro_dice: float
    foreground_macro_target: float
    class_dice: dict[str, float]
    class_targets: dict[str, float]
    class_passes: dict[str, bool]
    run_config: dict[str, str] | None = None


def _parse_dice(raw_value: str | None, class_name: str) -> float:
    if raw_value is None or raw_value.strip() == "":
        raise TargetEvidenceError(f"Missing Dice value for {class_name}")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise TargetEvidenceError(f"Invalid Dice value for {class_name}: {raw_value!r}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise TargetEvidenceError(f"Dice for {class_name} must be finite and within [0, 1]: {raw_value!r}")
    return value


def read_foreground_dice(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.is_file():
        raise TargetEvidenceError(f"Metrics CSV does not exist: {metrics_path}")

    found: dict[str, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = DictReader(handle)
        if reader.fieldnames is None or "class" not in reader.fieldnames or "dice" not in reader.fieldnames:
            raise TargetEvidenceError("Metrics CSV must contain 'class' and 'dice' columns")
        for row in reader:
            class_name = (row.get("class") or "").strip()
            if class_name not in PAPER_DICE_TARGETS:
                # Background and the existing four-class Mean are deliberately
                # ignored so they can never change the foreground decision.
                continue
            if class_name in found:
                raise TargetEvidenceError(f"Duplicate metrics row for {class_name}")
            found[class_name] = _parse_dice(row.get("dice"), class_name)

    missing = [name for name in PAPER_DICE_TARGETS if name not in found]
    if missing:
        raise TargetEvidenceError(f"Missing required class rows: {', '.join(missing)}")
    return found


def evaluate_target(
    class_dice: dict[str, float],
    foreground_macro_target: float = FOREGROUND_MACRO_TARGET,
) -> TargetCheck:
    missing = [name for name in PAPER_DICE_TARGETS if name not in class_dice]
    if missing:
        raise TargetEvidenceError(f"Missing required class scores: {', '.join(missing)}")

    normalized = {name: _parse_dice(str(class_dice[name]), name) for name in PAPER_DICE_TARGETS}
    macro = sum(normalized.values()) / len(PAPER_DICE_TARGETS)
    class_passes = {
        name: normalized[name] >= target
        for name, target in PAPER_DICE_TARGETS.items()
    }
    if not math.isfinite(foreground_macro_target) or not 0.0 <= foreground_macro_target <= 1.0:
        raise TargetEvidenceError("Foreground macro target must be finite and within [0, 1]")
    score_target_met = macro >= foreground_macro_target and all(class_passes.values())
    return TargetCheck(
        score_target_met=score_target_met,
        foreground_macro_dice=macro,
        foreground_macro_target=foreground_macro_target,
        class_dice=normalized,
        class_targets=dict(PAPER_DICE_TARGETS),
        class_passes=class_passes,
    )


def read_run_config(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        raise TargetEvidenceError(f"Run config does not exist: {config_path}")
    values: dict[str, str] = {}
    with config_path.open(newline="", encoding="utf-8") as handle:
        reader = DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["key", "value"]:
            raise TargetEvidenceError("Run config must contain tab-separated 'key' and 'value' columns")
        for row in reader:
            key = (row.get("key") or "").strip()
            if not key:
                raise TargetEvidenceError("Run config contains an empty key")
            if key in values:
                raise TargetEvidenceError(f"Run config contains duplicate key: {key}")
            values[key] = row.get("value") or ""
    return values


def check_metrics(
    metrics_path: Path,
    foreground_macro_target: float = FOREGROUND_MACRO_TARGET,
    run_config_path: Path | None = None,
    required_sequence: str | None = None,
) -> TargetCheck:
    check = evaluate_target(read_foreground_dice(metrics_path), foreground_macro_target)
    if required_sequence is not None and run_config_path is None:
        raise TargetEvidenceError("A run config is required when validating the MRI sequence")
    if run_config_path is not None:
        run_config = read_run_config(run_config_path)
        if required_sequence is not None and run_config.get("sequences") != required_sequence:
            raise TargetEvidenceError(
                f"Expected sequence {required_sequence!r}, got {run_config.get('sequences')!r}"
            )
        check = replace(check, run_config=run_config)
    return check


def write_result(output_path: Path, check: TargetCheck) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(check), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = ArgumentParser(description="Check the foreground T2 SPACE paper-reproduction target.")
    parser.add_argument("metrics", type=Path, help="Path to validation_metrics.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to target_check.json beside the metrics CSV.",
    )
    parser.add_argument("--foreground_macro_target", type=float, default=FOREGROUND_MACRO_TARGET)
    parser.add_argument("--run-config", type=Path, default=None, help="Optional run_config.tsv evidence.")
    parser.add_argument(
        "--require-sequence",
        default=None,
        help="Require run_config.tsv to contain this exact sequence, for example T2_SPACE.",
    )
    args = parser.parse_args()
    output_path = args.output or args.metrics.with_name("target_check.json")

    try:
        check = check_metrics(
            args.metrics,
            args.foreground_macro_target,
            run_config_path=args.run_config,
            required_sequence=args.require_sequence,
        )
    except TargetEvidenceError as exc:
        print(f"Invalid target evidence: {exc}", file=sys.stderr)
        return 2

    write_result(output_path, check)
    print(f"Foreground macro Dice: {check.foreground_macro_dice:.6f}")
    for class_name in ("IVDs", "Vertebrae", "Spinal Canal"):
        status = "PASS" if check.class_passes[class_name] else "MISS"
        print(
            f"{class_name}: {check.class_dice[class_name]:.6f} "
            f"(target {check.class_targets[class_name]:.4f}) {status}"
        )
    print(f"Score target met: {'YES' if check.score_target_met else 'NO'}")
    print(f"Wrote {output_path}")
    return 0 if check.score_target_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
