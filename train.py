from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from arguments import add_data_args, add_model_args, add_optimization_args, get_param_groups
from spine_baseline.dataset import create_dataset
from spine_baseline.file_lists import (
    exclude_files,
    read_file_list,
    validate_disjoint_cohorts,
    validate_slice_files,
)
from spine_baseline.losses import combined_loss
from spine_baseline.metrics import dice_coefficient, mean_iou
from spine_baseline.model import build_modified_unet
from spine_baseline.preprocessing import extract_slices, filter_slices, split_train_val


def prepare_output(output_root: Path) -> Path:
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def write_file_list(path: Path, files: list[str]) -> None:
    """Persist the exact cohort used by this run for reproducible evaluation."""
    path.write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Train the Ahmed et al. 2025 baseline Modified U-Net.")
    add_data_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    parser.add_argument(
        "--train_file_list",
        type=Path,
        default=None,
        help="Exact training cohort. Explicit lists may intentionally bypass the Overview split.",
    )
    parser.add_argument(
        "--validation_file_list",
        type=Path,
        default=None,
        help="Exact validation cohort for model selection and early stopping.",
    )
    parser.add_argument(
        "--run_output_root",
        type=Path,
        default=None,
        help="Optional directory for this run's cohorts, checkpoints, and logs; defaults to output_root.",
    )
    parser.add_argument(
        "--reuse_processed_only",
        action="store_true",
        help=(
            "Do not run preprocessing or filtering. Requires exact train and validation lists and "
            "fails if any existing image/mask pair is invalid. Intended for read-only batch probes."
        ),
    )
    args = parser.parse_args()
    data, model_params, opt = get_param_groups(args)

    np.random.seed(opt.seed)
    tf.random.set_seed(opt.seed)

    explicit_train_files = read_file_list(args.train_file_list) if args.train_file_list is not None else None
    explicit_val_files = (
        read_file_list(args.validation_file_list) if args.validation_file_list is not None else None
    )
    selected_files = None
    if explicit_train_files is not None or explicit_val_files is not None:
        # When fixed lists are supplied, extracting unrelated series turns a
        # one-step memory probe into a long preprocessing job. The extractor
        # understands exact slice names, so restrict work to the requested
        # union without changing the normal no-list path.
        selected_files = set(explicit_train_files or []) | set(explicit_val_files or [])

    if args.reuse_processed_only:
        if explicit_train_files is None or explicit_val_files is None:
            raise ValueError("--reuse_processed_only requires both explicit train and validation file lists")
        if data.force_reprocess:
            raise ValueError("--reuse_processed_only cannot be combined with --force_reprocess")
        extract_stats = {"mode": "reuse_processed_only", "files_processed": 0, "errors": []}
    else:
        extract_stats = extract_slices(
            data_root=data.data_root,
            output_root=data.output_root,
            target_height=data.target_height,
            target_width=data.target_width,
            sequences=data.sequences,
            force=data.force_reprocess,
            selected_files=selected_files,
            orientation_mode=data.orientation_mode,
            orientation_manifest=data.orientation_manifest,
        )
    if explicit_train_files is not None and explicit_val_files is not None:
        # A two-list smoke probe is already a complete cohort definition. Do not
        # run filter_slices here: besides wasting time, it would rewrite
        # filtered_files.txt in the shared preprocessing cache used by the full
        # experiment. The explicit lists are validated below before any GPU work.
        kept_files = [*explicit_train_files, *explicit_val_files]
        filter_stats = {
            "mode": "explicit_file_lists",
            "kept": len(kept_files),
            "kept_by_sequence": {},
        }
        train_files = explicit_train_files
        val_files = explicit_val_files
        unmatched = []
    else:
        kept_files, filter_stats = filter_slices(
            data.output_root,
            data.min_classes,
            data.imbalance_threshold,
            data.max_slices_per_sequence,
        )
        train_files, val_files, unmatched = split_train_val(data.data_root, kept_files)
    if explicit_train_files is not None:
        # A fixed smoke cohort is an explicit experiment input. It therefore
        # takes precedence over the Overview split just like a frozen validation
        # cohort, including when the selected series belongs to another split.
        train_files = explicit_train_files
    if explicit_val_files is not None:
        # Model selection is part of evaluation. Reusing the frozen cohort here
        # prevents a candidate filter from choosing its best epoch on an easier
        # validation set even when the final evaluator is correctly frozen.
        val_files = explicit_val_files

    if explicit_val_files is not None and explicit_train_files is None:
        # An explicitly frozen validation slice can intentionally come from a
        # series labelled "training" in Overview. Remove it from the derived
        # training cohort so the explicit choice wins without data leakage.
        train_files = exclude_files(train_files, explicit_val_files)
    if explicit_train_files is not None and explicit_val_files is None:
        val_files = exclude_files(val_files, explicit_train_files)

    if not train_files or not val_files:
        raise ValueError("Train/validation split is empty. Check data_root and filtered slice names.")

    allowed_sequences = (
        {item.strip().upper() for item in data.sequences.split(",") if item.strip()}
        if data.sequences else None
    )
    expected_shape = (data.target_height, data.target_width)
    validate_slice_files(train_files, data.output_root, "Train", allowed_sequences, expected_shape)
    validate_slice_files(val_files, data.output_root, "Validation", allowed_sequences, expected_shape)
    validate_disjoint_cohorts(train_files, val_files)

    # output_root remains the preprocessing cache for backward compatibility.
    # A separate run root is important for batch-size probes: smoke checkpoints
    # must never replace a full experiment's best model or evidence files.
    run_output_root = args.run_output_root or data.output_root

    # Filtering rules are experiment inputs, so recomputing the validation
    # cohort in evaluate.py could make an easier cohort look like a better
    # model. Save the exact lists before training and reuse validation_files.txt.
    run_output_root.mkdir(parents=True, exist_ok=True)
    write_file_list(run_output_root / "train_files.txt", train_files)
    write_file_list(run_output_root / "validation_files.txt", val_files)
    write_file_list(run_output_root / "unmatched_files.txt", unmatched)

    print("Extraction stats:", extract_stats)
    print("Filtering stats:", filter_stats)
    print(f"Train slices: {len(train_files)}")
    print(f"Validation slices: {len(val_files)}")
    print(f"Unmatched slices: {len(unmatched)}")

    train_ds = create_dataset(
        train_files, data.output_root, data.target_height, data.target_width,
        model_params.num_classes, opt.batch_size, shuffle=True,
    )
    val_ds = create_dataset(
        val_files, data.output_root, data.target_height, data.target_width,
        model_params.num_classes, opt.batch_size, shuffle=False,
    )

    model = build_modified_unet(
        input_shape=(data.target_height, data.target_width, model_params.input_channels),
        num_classes=model_params.num_classes,
        dropout_rate=model_params.dropout_rate,
        leaky_relu_alpha=model_params.leaky_relu_alpha,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=opt.learning_rate),
        loss=combined_loss(alpha=opt.focal_weight, gamma=opt.focal_gamma),
        metrics=["accuracy", mean_iou(model_params.num_classes), dice_coefficient(model_params.num_classes)],
    )

    checkpoint_dir = prepare_output(run_output_root)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mean_iou",
            mode="max",
            patience=opt.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_iou",
            mode="max",
            factor=0.5,
            patience=max(1, opt.patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(checkpoint_dir / "training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=opt.epochs, callbacks=callbacks, verbose=1)
    model.save(str(checkpoint_dir / "final_model.keras"))
    print(f"Training complete. Models and logs saved to: {checkpoint_dir}")
    print(f"History keys: {list(history.history.keys())}")


if __name__ == "__main__":
    main()
