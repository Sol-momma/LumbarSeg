from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from arguments import add_data_args, add_model_args, add_optimization_args, get_param_groups
from spine_baseline.dataset import create_dataset
from spine_baseline.losses import combined_loss
from spine_baseline.metrics import dice_coefficient, mean_iou
from spine_baseline.model import build_modified_unet
from spine_baseline.preprocessing import extract_slices, filter_slices, split_train_val


def prepare_output(output_root: Path) -> Path:
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def main() -> None:
    parser = ArgumentParser(description="Train the Ahmed et al. 2025 baseline Modified U-Net.")
    add_data_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    data, model_params, opt = get_param_groups(parser.parse_args())

    np.random.seed(opt.seed)
    tf.random.set_seed(opt.seed)

    extract_stats = extract_slices(
        data_root=data.data_root,
        output_root=data.output_root,
        target_height=data.target_height,
        target_width=data.target_width,
        sequences=data.sequences,
        force=data.force_reprocess,
    )
    kept_files, filter_stats = filter_slices(data.output_root, data.min_classes, data.imbalance_threshold)
    train_files, val_files, unmatched = split_train_val(data.data_root, kept_files)

    if not train_files or not val_files:
        raise ValueError("Train/validation split is empty. Check data_root and filtered slice names.")

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

    checkpoint_dir = prepare_output(data.output_root)
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
