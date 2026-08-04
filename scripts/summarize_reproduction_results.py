from __future__ import annotations

from argparse import ArgumentParser
from csv import DictReader
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from check_reproduction_target import PAPER_DICE_TARGETS, evaluate_target


@dataclass
class ExperimentResult:
    name: str
    metrics_path: Path
    training_log_path: Path | None
    mean_dice: float
    foreground_macro_dice: float | None
    score_target_met: bool
    t2_space_scope_verified: bool
    mean_iou: float
    vertebrae_dice: float | None
    canal_dice: float | None
    ivd_dice: float | None
    best_epoch: int | None
    best_val_mean_iou: float | None
    best_val_dice: float | None


def parse_float(value: str | None) -> float | None:
    if value is None or value == "" or value.lower() == "inf":
        return None
    return float(value)


def read_metrics(metrics_path: Path) -> dict[str, dict[str, str]]:
    with metrics_path.open(newline="") as handle:
        return {row["class"]: row for row in DictReader(handle)}


def read_training_best(training_log_path: Path | None) -> tuple[int | None, float | None, float | None]:
    if training_log_path is None or not training_log_path.exists():
        return None, None, None

    best_row: dict[str, str] | None = None
    best_iou = float("-inf")
    with training_log_path.open(newline="") as handle:
        for row in DictReader(handle):
            val_mean_iou = parse_float(row.get("val_mean_iou"))
            if val_mean_iou is not None and val_mean_iou > best_iou:
                best_iou = val_mean_iou
                best_row = row

    if best_row is None:
        return None, None, None
    return (
        int(float(best_row["epoch"])) + 1,
        best_iou,
        parse_float(best_row.get("val_dice_coefficient")),
    )


def read_run_sequence(metrics_path: Path) -> str | None:
    config_path = metrics_path.parent / "run_config.tsv"
    if not config_path.exists():
        return None
    with config_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["key"]: row["value"] for row in DictReader(handle, delimiter="\t")}
    return rows.get("sequences")


def discover_results(experiments_root: Path, target_dice: float = 0.97) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for metrics_path in sorted(experiments_root.glob("*/validation_metrics.csv")):
        rows = read_metrics(metrics_path)
        mean = rows.get("Mean")
        if mean is None:
            continue

        # Reproduction decisions should be based on the explicit class-wise
        # evaluator rather than the Keras training aggregate. The evaluator is
        # less convenient to compute, but it gives us the same per-class table
        # we need when explaining where the reproduction gap remains.
        training_log_path = metrics_path.parent / "training_log.csv"
        best_epoch, best_val_mean_iou, best_val_dice = read_training_best(training_log_path)
        vertebrae_dice = parse_float(rows.get("Vertebrae", {}).get("dice"))
        canal_dice = parse_float(rows.get("Spinal Canal", {}).get("dice"))
        ivd_dice = parse_float(rows.get("IVDs", {}).get("dice"))
        foreground_scores = {
            "Vertebrae": vertebrae_dice,
            "Spinal Canal": canal_dice,
            "IVDs": ivd_dice,
        }
        if all(value is not None for value in foreground_scores.values()):
            target_check = evaluate_target(
                {name: float(value) for name, value in foreground_scores.items()},
                foreground_macro_target=target_dice,
            )
            foreground_macro_dice = target_check.foreground_macro_dice
            score_target_met = target_check.score_target_met
        else:
            foreground_macro_dice = None
            score_target_met = False

        results.append(
            ExperimentResult(
                name=metrics_path.parent.name,
                metrics_path=metrics_path,
                training_log_path=training_log_path if training_log_path.exists() else None,
                mean_dice=float(mean["dice"]),
                foreground_macro_dice=foreground_macro_dice,
                score_target_met=score_target_met,
                t2_space_scope_verified=read_run_sequence(metrics_path) == "T2_SPACE",
                mean_iou=float(mean["iou"]),
                vertebrae_dice=vertebrae_dice,
                canal_dice=canal_dice,
                ivd_dice=ivd_dice,
                best_epoch=best_epoch,
                best_val_mean_iou=best_val_mean_iou,
                best_val_dice=best_val_dice,
            )
        )
    # The old four-class Mean includes Background and can overstate progress on
    # the anatomy that matters. Campaign ranking now follows the agreed
    # foreground-only macro while the historical Mean remains visible.
    return sorted(
        results,
        key=lambda item: item.foreground_macro_dice if item.foreground_macro_dice is not None else float("-inf"),
        reverse=True,
    )


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def make_recommendation(results: list[ExperimentResult], target_dice: float) -> str:
    if not results:
        return "No recorded validation metrics were found. Run a reproduction preset first."

    verified_pass = next(
        (result for result in results if result.score_target_met and result.t2_space_scope_verified),
        None,
    )
    if verified_pass is not None:
        return (
            f"A verified T2 SPACE foreground macro Dice is {verified_pass.foreground_macro_dice:.4f}, and every "
            "foreground class meets its paper value with a T2 SPACE manifest. The campaign target is met."
        )
    best = results[0]
    if best.score_target_met:
        return (
            f"Best recorded foreground macro Dice is {best.foreground_macro_dice:.4f}, and the numeric "
            "score target is met, but T2 SPACE scope is not verified by a run manifest."
        )
    return (
        f"Best recorded foreground macro Dice is {fmt(best.foreground_macro_dice)}, below the target "
        f"{target_dice:.4f}, or at least one class is below its paper floor. "
        "Run the next controlled T2 SPACE candidate in the goal campaign."
    )


def write_markdown(results: list[ExperimentResult], output_path: Path, target_dice: float) -> None:
    lines = [
        "# Reproduction Status",
        "",
        f"Updated: {date.today().isoformat()}",
        f"Target foreground macro Dice: `{target_dice:.4f}`",
        (
            "Class floors: "
            f"IVDs `{PAPER_DICE_TARGETS['IVDs']:.4f}`, "
            f"Vertebrae `{PAPER_DICE_TARGETS['Vertebrae']:.4f}`, "
            f"Spinal Canal `{PAPER_DICE_TARGETS['Spinal Canal']:.4f}`"
        ),
        "",
        "## Summary",
        "",
        "| Experiment | Foreground Macro Dice | 4-Class Mean Dice | Mean IoU | Vertebrae Dice | Canal Dice | IVD Dice | Score Target | T2 SPACE Scope | Best Epoch | Train Val Dice |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.name} | "
            f"{fmt(result.foreground_macro_dice)} | "
            f"{fmt(result.mean_dice)} | "
            f"{fmt(result.mean_iou)} | "
            f"{fmt(result.vertebrae_dice)} | "
            f"{fmt(result.canal_dice)} | "
            f"{fmt(result.ivd_dice)} | "
            f"{'PASS' if result.score_target_met else 'MISS'} | "
            f"{'VERIFIED' if result.t2_space_scope_verified else 'UNVERIFIED'} | "
            f"{result.best_epoch if result.best_epoch is not None else '-'} | "
            f"{fmt(result.best_val_dice)} |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
        make_recommendation(results, target_dice),
        "",
        "## Notes",
        "",
        "- `Foreground Macro Dice` averages IVDs, Vertebrae, and Spinal Canal only.",
        "- `4-Class Mean Dice` is the historical evaluator row and includes Background; it is not the success oracle.",
        "- `Score Target PASS` requires the foreground macro target and all three paper class floors.",
        "- Campaign success additionally requires `T2 SPACE Scope VERIFIED` from `run_config.tsv`.",
        "- `Train Val Dice` comes from the Keras training CSV and is secondary for paper comparison.",
        "- Mixed-sequence results are diagnostic and must not replace the T2 SPACE success definition.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Summarize recorded LumbarSeg reproduction metrics.")
    parser.add_argument("--experiments_root", type=Path, default=Path("docs/experiments"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiments/reproduction_status.md"))
    parser.add_argument("--target_dice", type=float, default=0.97)
    args = parser.parse_args()

    results = discover_results(args.experiments_root, args.target_dice)
    write_markdown(results, args.output, args.target_dice)
    print(f"Wrote {args.output}")
    print(make_recommendation(results, args.target_dice))


if __name__ == "__main__":
    main()
