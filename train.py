from argparse import ArgumentParser
import os
from pathlib import Path

# TensorFlow reads this switch while initializing devices. Set it before the
# import so a run cannot silently fall back to non-deterministic GPU kernels.
# The seed is still supplied below because deterministic kernels alone do not
# control initialization or dataset shuffling.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow import keras

from arguments import add_data_args, add_model_args, add_optimization_args, get_param_groups
from spine_baseline.class_weights import (
    derive_focal_class_weights,
    write_focal_class_weight_evidence,
)
from spine_baseline.dataset import create_dataset
from spine_baseline.file_lists import (
    COHORT_DISJOINT_MODES,
    COHORT_MODE_STRICT_SERIES,
    exclude_files,
    read_file_list,
    validate_disjoint_cohorts,
    validate_slice_files,
    write_cohort_validation_report,
)
from spine_baseline.losses import combined_loss, validate_loss_configuration, write_loss_config
from spine_baseline.metrics import dice_coefficient, mean_iou
from spine_baseline.model import build_modified_unet
from spine_baseline.preprocessing import extract_slices, filter_slices, split_train_val
from spine_baseline.training_resume import (
    read_resume_best_metric,
    validate_training_resume_state,
    write_training_resume_evidence,
)


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
    parser.add_argument(
        "--reuse_processed_cache",
        action="store_true",
        help=(
            "Read an existing processed cache without extraction, then derive this run's training "
            "cohort with its filtering settings. Requires --run_output_root and a fixed validation list."
        ),
    )
    parser.add_argument(
        "--cohort_disjoint_mode",
        choices=COHORT_DISJOINT_MODES,
        default=COHORT_MODE_STRICT_SERIES,
        help=(
            "Cohort isolation policy. strict_series is required for final generalization evidence. "
            "author_diagnostic_slice permits one MRI series across cohorts only for paper-alignment diagnosis."
        ),
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help=(
            "Resume an interrupted explicit run from its Keras epoch backup. "
            "Requires --run_output_root and exact saved train/validation cohorts."
        ),
    )
    args = parser.parse_args()
    data, model_params, opt = get_param_groups(args)
    # Fail before reading or writing experiment data when the requested loss
    # would combine two causal changes or use an invalid negative boost.
    validate_loss_configuration(
        opt.focal_class_weight_mode,
        opt.focal_canal_boundary_boost,
    )

    # set_random_seed covers Python, NumPy, and TensorFlow. Explicitly enabling
    # deterministic ops makes the recorded seed meaningful on GPU rather than
    # merely making initialization repeatable.
    keras.utils.set_random_seed(opt.seed)
    tf.config.experimental.enable_op_determinism()

    explicit_train_files = read_file_list(args.train_file_list) if args.train_file_list is not None else None
    explicit_val_files = (
        read_file_list(args.validation_file_list) if args.validation_file_list is not None else None
    )
    run_output_root = args.run_output_root or data.output_root
    if args.resume_training and args.run_output_root is None:
        raise ValueError("--resume_training requires an explicit --run_output_root")
    if args.reuse_processed_cache and args.run_output_root is None:
        raise ValueError("--reuse_processed_cache requires an explicit --run_output_root")
    if args.reuse_processed_cache and args.reuse_processed_only:
        raise ValueError("Choose either --reuse_processed_cache or --reuse_processed_only")
    selected_files = None
    if explicit_train_files is not None and explicit_val_files is not None:
        # When both fixed lists are supplied, extracting unrelated series turns a
        # one-step memory probe into a long preprocessing job. The extractor
        # understands exact slice names, so restrict work to the requested
        # union without changing the normal no-list path.
        selected_files = set(explicit_train_files or []) | set(explicit_val_files or [])

    if args.reuse_processed_only or args.reuse_processed_cache:
        if args.reuse_processed_only and (explicit_train_files is None or explicit_val_files is None):
            raise ValueError("--reuse_processed_only requires both explicit train and validation file lists")
        if args.reuse_processed_cache and explicit_val_files is None:
            raise ValueError("--reuse_processed_cache requires an explicit validation file list")
        if data.force_reprocess:
            raise ValueError("Processed-cache reuse cannot be combined with --force_reprocess")
        for required_dir in (data.output_root / "images", data.output_root / "masks"):
            if not required_dir.is_dir():
                raise ValueError(f"Processed cache directory does not exist: {required_dir}")
        extract_stats = {
            "mode": "reuse_processed_only" if args.reuse_processed_only else "reuse_processed_cache",
            "files_processed": 0,
            "errors": [],
        }
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
            evidence_root=run_output_root,
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
    cohort_report = validate_disjoint_cohorts(
        train_files,
        val_files,
        mode=args.cohort_disjoint_mode,
    )
    if cohort_report.warning:
        print(f"WARNING: {cohort_report.warning}")

    # output_root remains the preprocessing cache for backward compatibility.
    # A separate run root is important for batch-size probes: smoke checkpoints
    # must never replace a full experiment's best model or evidence files.
    resume_paths = None
    if args.run_output_root is not None:
        resume_paths = validate_training_resume_state(
            run_output_root,
            resume_requested=args.resume_training,
            train_files=train_files,
            validation_files=val_files,
        )

    # Filtering rules are experiment inputs, so recomputing the validation
    # cohort in evaluate.py could make an easier cohort look like a better
    # model. Save the exact lists before training and reuse validation_files.txt.
    run_output_root.mkdir(parents=True, exist_ok=True)
    write_cohort_validation_report(run_output_root / "cohort_validation.tsv", cohort_report)
    write_file_list(run_output_root / "train_files.txt", train_files)
    write_file_list(run_output_root / "validation_files.txt", val_files)
    write_file_list(run_output_root / "unmatched_files.txt", unmatched)

    focal_class_weights, focal_class_counts = derive_focal_class_weights(
        opt.focal_class_weight_mode,
        train_files,
        data.output_root,
        model_params.num_classes,
    )
    write_focal_class_weight_evidence(
        run_output_root / "focal_class_weights.tsv",
        opt.focal_class_weight_mode,
        focal_class_counts,
        focal_class_weights,
    )
    write_loss_config(
        run_output_root / "loss_config.tsv",
        focal_weight=opt.focal_weight,
        focal_gamma=opt.focal_gamma,
        focal_class_weight_mode=opt.focal_class_weight_mode,
        focal_canal_boundary_boost=opt.focal_canal_boundary_boost,
    )

    print("Extraction stats:", extract_stats)
    print("Filtering stats:", filter_stats)
    print(f"Train slices: {len(train_files)}")
    print(f"Validation slices: {len(val_files)}")
    print(f"Unmatched slices: {len(unmatched)}")
    print(
        "Focal class weights:",
        "equal" if focal_class_weights is None else focal_class_weights.tolist(),
    )
    print("Focal canal boundary boost:", opt.focal_canal_boundary_boost)

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
    # Deterministic GPU execution and Keras' automatic XLA path are not
    # compatible here: TensorFlow 2.15 has no deterministic XLA MaxPool
    # gradient, and the boundary candidate also exceeds the available 8 GiB
    # when XLA is selected. Keep one auditable execution path for every
    # candidate instead of allowing runtime-dependent "auto" compilation.
    jit_compile = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=opt.learning_rate),
        loss=combined_loss(
            alpha=opt.focal_weight,
            gamma=opt.focal_gamma,
            class_weights=focal_class_weights,
            canal_boundary_boost=opt.focal_canal_boundary_boost,
        ),
        metrics=["accuracy", mean_iou(model_params.num_classes), dice_coefficient(model_params.num_classes)],
        jit_compile=jit_compile,
    )

    checkpoint_dir = prepare_output(run_output_root)
    checkpoint_threshold = None
    if args.resume_training:
        checkpoint_threshold = read_resume_best_metric(
            checkpoint_dir / "training_log.csv",
            "val_mean_iou",
        )
    write_training_resume_evidence(
        run_output_root / "training_resume.tsv",
        resume_requested=args.resume_training,
        historical_best=checkpoint_threshold,
    )
    callbacks = []
    if resume_paths is not None:
        # BackupAndRestore persists the model, optimizer, and completed epoch.
        # It does not persist EarlyStopping or ReduceLROnPlateau counters. Its
        # automatic restore is guarded above by an explicit flag and exact
        # cohort comparison, and training_resume.tsv keeps a resumed run out of
        # the canonical uninterrupted comparison.
        callbacks.append(
            keras.callbacks.BackupAndRestore(
                backup_dir=str(resume_paths.backup_dir),
                save_freq="epoch",
                delete_checkpoint=True,
            )
        )
    callbacks.extend([
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            initial_value_threshold=checkpoint_threshold,
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
        keras.callbacks.CSVLogger(
            str(checkpoint_dir / "training_log.csv"),
            append=args.resume_training,
        ),
    ])

    history = model.fit(train_ds, validation_data=val_ds, epochs=opt.epochs, callbacks=callbacks, verbose=1)
    model.save(str(checkpoint_dir / "final_model.keras"))
    print(f"Training complete. Models and logs saved to: {checkpoint_dir}")
    print(f"History keys: {list(history.history.keys())}")


if __name__ == "__main__":
    main()
