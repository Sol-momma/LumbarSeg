from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataParams:
    data_root: Path
    output_root: Path
    sequences: str | None
    target_height: int
    target_width: int
    min_classes: int
    imbalance_threshold: float
    max_slices_per_sequence: int | None
    force_reprocess: bool
    orientation_mode: str
    orientation_manifest: Path | None


@dataclass
class ModelParams:
    input_channels: int
    num_classes: int
    dropout_rate: float | None
    leaky_relu_alpha: float


@dataclass
class OptimizationParams:
    batch_size: int
    epochs: int
    learning_rate: float
    focal_weight: float
    focal_gamma: float
    focal_class_weight_mode: str
    patience: int
    seed: int


def add_data_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("Data Parameters")
    group.add_argument(
        "--data_root",
        type=Path,
        default=Path("/content/drive/MyDrive/SPIDER/DataSet"),
        help="SPIDER dataset root containing images/, masks/, and overview CSV.",
    )
    group.add_argument(
        "--output_root",
        type=Path,
        default=Path("/content/drive/MyDrive/SPIDER/processed_baseline"),
        help="Directory where preprocessed slices and metadata are written.",
    )
    group.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Optional comma-separated sequence filter: T1,T2,T2_SPACE.",
    )
    group.add_argument("--target_height", type=int, default=512)
    group.add_argument("--target_width", type=int, default=640)
    group.add_argument("--min_classes", type=int, default=4)
    group.add_argument("--imbalance_threshold", type=float, default=0.55)
    group.add_argument(
        "--max_slices_per_sequence",
        type=int,
        default=1000,
        help="Maximum kept slices per MRI sequence after filtering. Use 0 to disable the paper's 1000-slice cap.",
    )
    group.add_argument("--force_reprocess", action="store_true")
    group.add_argument(
        "--orientation_mode",
        choices=("legacy", "metadata", "manifest"),
        default="legacy",
        help=(
            "Sagittal orientation policy. 'legacy' preserves the smallest-axis baseline; "
            "'metadata' uses SimpleITK direction; 'manifest' requires reviewed per-series transforms."
        ),
    )
    group.add_argument(
        "--orientation_manifest",
        type=Path,
        default=None,
        help="Reviewed orientation CSV required by --orientation_mode manifest.",
    )


def add_model_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("Model Parameters")
    group.add_argument("--input_channels", type=int, default=1)
    group.add_argument("--num_classes", type=int, default=4)
    group.add_argument(
        "--dropout_rate",
        type=float,
        default=None,
        help="Optional global dropout override. Omit to use the paper schedule: 0.1/0.2/0.3 by depth.",
    )
    group.add_argument("--leaky_relu_alpha", type=float, default=0.1)


def add_optimization_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("Optimization Parameters")
    group.add_argument("--batch_size", type=int, default=8)
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--learning_rate", type=float, default=1e-4)
    group.add_argument("--focal_weight", type=float, default=0.6)
    group.add_argument("--focal_gamma", type=float, default=4.0)
    group.add_argument(
        "--focal_class_weight_mode",
        choices=("none", "inverse_sqrt_train"),
        default="none",
        help=(
            "Optional focal-loss class weighting. inverse_sqrt_train derives stable weights "
            "from training masks only and records them in the run output."
        ),
    )
    group.add_argument("--patience", type=int, default=15)
    group.add_argument("--seed", type=int, default=42)


def get_data_params(args: Namespace) -> DataParams:
    return DataParams(
        data_root=args.data_root,
        output_root=args.output_root,
        sequences=args.sequences,
        target_height=args.target_height,
        target_width=args.target_width,
        min_classes=args.min_classes,
        imbalance_threshold=args.imbalance_threshold,
        max_slices_per_sequence=args.max_slices_per_sequence if args.max_slices_per_sequence > 0 else None,
        force_reprocess=args.force_reprocess,
        orientation_mode=args.orientation_mode,
        orientation_manifest=args.orientation_manifest,
    )


def get_param_groups(args: Namespace) -> tuple[DataParams, ModelParams, OptimizationParams]:
    data = get_data_params(args)
    model = ModelParams(
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        leaky_relu_alpha=args.leaky_relu_alpha,
    )
    opt = OptimizationParams(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        focal_weight=args.focal_weight,
        focal_gamma=args.focal_gamma,
        focal_class_weight_mode=args.focal_class_weight_mode,
        patience=args.patience,
        seed=args.seed,
    )
    return data, model, opt
