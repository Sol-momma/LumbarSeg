from __future__ import annotations

from argparse import ArgumentParser
from csv import DictReader
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class ExperimentResult:
    name: str
    metrics_path: Path
    training_log_path: Path | None
    mean_dice: float
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


def discover_results(experiments_root: Path) -> list[ExperimentResult]:
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
        results.append(
            ExperimentResult(
                name=metrics_path.parent.name,
                metrics_path=metrics_path,
                training_log_path=training_log_path if training_log_path.exists() else None,
                mean_dice=float(mean["dice"]),
                mean_iou=float(mean["iou"]),
                vertebrae_dice=parse_float(rows.get("Vertebrae", {}).get("dice")),
                canal_dice=parse_float(rows.get("Spinal Canal", {}).get("dice")),
                ivd_dice=parse_float(rows.get("IVDs", {}).get("dice")),
                best_epoch=best_epoch,
                best_val_mean_iou=best_val_mean_iou,
                best_val_dice=best_val_dice,
            )
        )
    return sorted(results, key=lambda item: item.mean_dice, reverse=True)


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def make_recommendation(results: list[ExperimentResult], target_dice: float) -> str:
    if not results:
        return "No recorded validation metrics were found. Run a reproduction preset first."

    best = results[0]
    has_all_sequences = any(result.name.startswith("all_sequences") for result in results)
    if best.mean_dice >= target_dice:
        return (
            f"Best recorded Mean Dice is {best.mean_dice:.4f}, which meets the target "
            f"{target_dice:.2f}. Start improvement experiments only after confirming the paper metric definition."
        )
    if not has_all_sequences:
        return (
            "Run `all_4cls090_cap1000` next. The current recorded results do not yet include "
            "the combined T1/T2/T2_SPACE condition, so the reproduction baseline is incomplete."
        )
    return (
        f"Best recorded Mean Dice is {best.mean_dice:.4f}, below the target {target_dice:.2f}. "
        "Keep reproduction work active before treating improvements as comparable to the paper."
    )


def write_markdown(results: list[ExperimentResult], output_path: Path, target_dice: float) -> None:
    lines = [
        "# Reproduction Status",
        "",
        f"Updated: {date.today().isoformat()}",
        f"Target paper-level Dice: `{target_dice:.2f}`",
        "",
        "## Summary",
        "",
        "| Experiment | Mean Dice | Mean IoU | Vertebrae Dice | Canal Dice | IVD Dice | Best Epoch | Train Val Dice |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.name} | "
            f"{fmt(result.mean_dice)} | "
            f"{fmt(result.mean_iou)} | "
            f"{fmt(result.vertebrae_dice)} | "
            f"{fmt(result.canal_dice)} | "
            f"{fmt(result.ivd_dice)} | "
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
        "- `Mean Dice` comes from `evaluate.py` class-wise validation metrics.",
        "- `Train Val Dice` comes from the Keras training CSV and is secondary for paper comparison.",
        "- Missing all-sequence results mean the reproduction baseline should still be treated as incomplete.",
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

    results = discover_results(args.experiments_root)
    write_markdown(results, args.output, args.target_dice)
    print(f"Wrote {args.output}")
    print(make_recommendation(results, args.target_dice))


if __name__ == "__main__":
    main()
