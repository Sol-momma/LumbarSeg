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
PAPER_PROTOCOL_STATUSES = ("verified", "unverified", "blocked")
PAPER_PROTOCOL_ASPECTS = (
    "preprocessing",
    "filtering",
    "split",
    "training",
    "evaluation",
)


class TargetEvidenceError(ValueError):
    """Raised when a metrics CSV cannot support a reproducible decision."""


@dataclass(frozen=True)
class TargetCheck:
    score_target_met: bool
    paper_class_targets_met: bool
    foreground_macro_target_met: bool
    foreground_macro_dice: float
    foreground_macro_target: float
    class_dice: dict[str, float]
    class_targets: dict[str, float]
    class_passes: dict[str, bool]
    paper_protocol_verified: str
    paper_protocol_evidence: list[str]
    uninterrupted_run_equivalent: bool | None = None
    training_resume_evidence: dict[str, str] | None = None
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
    paper_class_targets_met = all(class_passes.values())
    foreground_macro_target_met = macro >= foreground_macro_target
    # Keep the established combined campaign oracle for callers that already
    # consume it, while exposing its two independent decisions. The exact paper
    # class vector averages 0.969033, so class reproduction and the project's
    # separate 0.9700 macro ambition must never be reported as one condition.
    score_target_met = foreground_macro_target_met and paper_class_targets_met
    return TargetCheck(
        score_target_met=score_target_met,
        paper_class_targets_met=paper_class_targets_met,
        foreground_macro_target_met=foreground_macro_target_met,
        foreground_macro_dice=macro,
        foreground_macro_target=foreground_macro_target,
        class_dice=normalized,
        class_targets=dict(PAPER_DICE_TARGETS),
        class_passes=class_passes,
        paper_protocol_verified="unverified",
        paper_protocol_evidence=["run_config.tsv was not supplied; paper protocol cannot be verified."],
    )


def assess_paper_protocol(run_config: dict[str, str] | None) -> tuple[str, list[str]]:
    """Return a three-state paper-protocol decision with inspectable evidence.

    A numeric Dice match cannot verify protocol alignment. Every material stage
    must be explicitly marked and supported by evidence in run_config.tsv.
    Known proxy or diagnostic modes prevent an accidental promotion to verified.
    """
    if run_config is None:
        return (
            "unverified",
            ["run_config.tsv was not supplied; paper protocol cannot be verified."],
        )

    evidence: list[str] = []
    has_blocker = False
    has_unverified = False

    sequence = run_config.get("sequences", "").strip()
    if not sequence:
        has_unverified = True
        evidence.append("MRI sequence is missing from run_config.tsv.")
    elif sequence != "T2_SPACE":
        has_blocker = True
        evidence.append(f"MRI sequence is {sequence!r}, not the paper comparison scope 'T2_SPACE'.")
    else:
        evidence.append("MRI sequence is recorded as T2_SPACE.")

    filter_definition = run_config.get("filter_definition", "").strip()
    if filter_definition.startswith("paper_filter_proxy"):
        has_blocker = True
        evidence.append(
            "Filtering uses a documented paper_filter_proxy; the unpublished exact 55% rule is blocked."
        )

    diagnostic_mode = (
        run_config.get("split_mode", "").startswith("author_diagnostic_")
        or run_config.get("cohort_disjoint_mode") == "author_diagnostic_slice"
        or run_config.get("final_generalization_evidence", "").casefold() == "false"
    )
    if diagnostic_mode:
        has_unverified = True
        evidence.append(
            "Author-style slice splitting is diagnostic only and is not final generalization evidence."
        )

    for aspect in PAPER_PROTOCOL_ASPECTS:
        status_key = f"paper_protocol_{aspect}_status"
        evidence_key = f"paper_protocol_{aspect}_evidence"
        status = run_config.get(status_key, "").strip().casefold()
        aspect_evidence = run_config.get(evidence_key, "").strip()
        if not status:
            has_unverified = True
            evidence.append(f"{aspect}: status is missing.")
            continue
        if status not in PAPER_PROTOCOL_STATUSES:
            raise TargetEvidenceError(
                f"{status_key} must be one of {', '.join(PAPER_PROTOCOL_STATUSES)}; got {status!r}"
            )
        if not aspect_evidence:
            has_unverified = True
            evidence.append(f"{aspect}: {status} was declared without supporting evidence.")
        else:
            evidence.append(f"{aspect}: {status} - {aspect_evidence}")
        if status == "blocked":
            has_blocker = True
        elif status == "unverified":
            has_unverified = True

    if has_blocker:
        return "blocked", evidence
    if has_unverified:
        return "unverified", evidence
    return "verified", evidence


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
    training_resume_path: Path | None = None,
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
        protocol_status, protocol_evidence = assess_paper_protocol(run_config)
        check = replace(
            check,
            run_config=run_config,
            paper_protocol_verified=protocol_status,
            paper_protocol_evidence=protocol_evidence,
        )
    if training_resume_path is not None:
        resume_evidence = read_run_config(training_resume_path)
        raw_equivalence = resume_evidence.get("equivalent_to_uninterrupted_run", "").casefold()
        if raw_equivalence not in {"true", "false"}:
            raise TargetEvidenceError(
                "Training resume evidence must declare equivalent_to_uninterrupted_run as true or false"
            )
        check = replace(
            check,
            uninterrupted_run_equivalent=raw_equivalence == "true",
            training_resume_evidence=resume_evidence,
        )
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
        "--training-resume",
        type=Path,
        default=None,
        help="Optional training_resume.tsv evidence for interrupted-run equivalence.",
    )
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
            training_resume_path=args.training_resume,
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
    print(f"Paper class values reproduced: {'YES' if check.paper_class_targets_met else 'NO'}")
    print(
        f"Foreground macro target {check.foreground_macro_target:.4f} met: "
        f"{'YES' if check.foreground_macro_target_met else 'NO'}"
    )
    print(f"Combined campaign score target met: {'YES' if check.score_target_met else 'NO'}")
    print(f"Paper protocol: {check.paper_protocol_verified.upper()}")
    for evidence in check.paper_protocol_evidence:
        print(f"  - {evidence}")
    if check.uninterrupted_run_equivalent is not None:
        print(
            "Equivalent to an uninterrupted run: "
            f"{'YES' if check.uninterrupted_run_equivalent else 'NO'}"
        )
    print(f"Wrote {output_path}")
    return 0 if check.score_target_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
