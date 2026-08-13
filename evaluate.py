from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf

from arguments import add_data_args, add_model_args, add_optimization_args, get_param_groups
from spine_baseline.file_lists import read_file_list
from spine_baseline.losses import combined_loss
from spine_baseline.metrics import dice_coefficient, evaluate_classwise_with_aggregations, mean_iou
from spine_baseline.preprocessing import filter_slices, split_train_val


def main() -> None:
    parser = ArgumentParser(description="Evaluate a trained baseline model.")
    add_data_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    parser.add_argument("--model_path", required=True, help="Path to a .keras model.")
    parser.add_argument(
        "--evaluation_output_root",
        type=Path,
        default=None,
        help=(
            "Optional directory for evaluation CSVs. Defaults to output_root for backward "
            "compatibility; set this when output_root is a shared read-only processed cache."
        ),
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help="Exact validation slice list. When omitted, the cohort is recomputed for backward compatibility.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional validation slice limit for quick checks.")
    parser.add_argument(
        "--nsd_tolerance",
        type=float,
        default=1.0,
        help="Surface-distance tolerance for NSD. Uses mm when preprocessed spacing metadata is available.",
    )
    args = parser.parse_args()
    data, model_params, opt = get_param_groups(args)

    if args.file_list is not None:
        # Goal-campaign comparisons must use one frozen validation cohort even
        # when a candidate changes the training filter or sampling strategy.
        val_files = read_file_list(args.file_list)
    else:
        kept_files, _ = filter_slices(
            data.output_root,
            data.min_classes,
            data.imbalance_threshold,
            data.max_slices_per_sequence,
        )
        _, val_files, _ = split_train_val(data.data_root, kept_files)

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            "loss_fn": combined_loss(alpha=opt.focal_weight, gamma=opt.focal_gamma),
            "mean_iou": mean_iou(model_params.num_classes),
            "dice_coefficient": dice_coefficient(model_params.num_classes),
        },
        compile=False,
    )
    results, aggregation_results = evaluate_classwise_with_aggregations(
        model,
        val_files,
        data.output_root,
        model_params.num_classes,
        limit=args.limit,
        nsd_tolerance=args.nsd_tolerance,
    )
    evaluation_output_root = args.evaluation_output_root or data.output_root
    evaluation_output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = evaluation_output_root / "validation_metrics.csv"
    results.to_csv(metrics_path, index=False)
    # Keep the historical CSV byte-shape compatible for goal checks and place
    # alternative aggregation definitions in an explicitly separate artifact.
    aggregations_path = evaluation_output_root / "validation_metrics_aggregations.csv"
    aggregation_results.to_csv(aggregations_path, index=False)
    print(results)
    print(f"Saved metrics to: {metrics_path}")
    print(aggregation_results)
    print(f"Saved metric aggregations to: {aggregations_path}")


if __name__ == "__main__":
    main()
