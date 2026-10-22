from pathlib import Path

import numpy as np
import tensorflow as tf


def load_sample(filename: str, output_root: Path, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.load(output_root / "images" / filename)["image"].astype(np.float32)
    mask = np.load(output_root / "masks" / filename)["mask"].astype(np.int32)

    image_min, image_max = image.min(), image.max()
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    image = image[..., np.newaxis]
    mask_onehot = np.eye(num_classes, dtype=np.float32)[mask]
    return image, mask_onehot


def load_slice_spacing(filename: str, output_root: Path) -> np.ndarray:
    with np.load(output_root / "images" / filename) as sample:
        if "spacing" in sample:
            return sample["spacing"].astype(np.float32)
    return np.array([1.0, 1.0], dtype=np.float32)


AUGMENT_LEVELS = ("none", "strong")


def augment_sample(image: np.ndarray, mask_onehot: np.ndarray, rng: np.random.Generator,
                   level: str = "strong") -> tuple[np.ndarray, np.ndarray]:
    """Training-only augmentation. "strong": random rotation +-15 deg, scale 0.85-1.15, shift +-8 %,
    gamma 0.7-1.5, brightness/contrast +-10 %, Gaussian noise (sd 0-0.03), Gaussian blur (p=0.2).
    Masks use nearest-neighbour warping so labels stay one-hot. No flips (sagittal anatomy is oriented)."""
    if level == "none":
        return image, mask_onehot
    if level not in AUGMENT_LEVELS:
        raise ValueError(f"unknown augment level {level!r}; choose one of {AUGMENT_LEVELS}")
    import cv2
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.85, 1.15))
    m[:, 2] += (rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h)
    img = cv2.warpAffine(image[..., 0], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lab = cv2.warpAffine(np.argmax(mask_onehot, -1).astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = np.clip(img, 0, 1) ** rng.uniform(0.7, 1.5)
    img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.0))
    img = img + rng.normal(0, rng.uniform(0, 0.03), img.shape)
    img = np.clip(img, 0, 1).astype(np.float32)[..., np.newaxis]
    return img, np.eye(mask_onehot.shape[-1], dtype=np.float32)[lab]


def create_dataset(file_list: list[str], output_root: Path, target_height: int, target_width: int,
                   num_classes: int, batch_size: int, shuffle: bool, augment: str = "none",
                   seed: int = 42, augment_workers: int = 1) -> tf.data.Dataset:
    if not file_list:
        raise ValueError("file_list is empty")
    if augment != "none" and augment_workers > 1:
        return _create_parallel_augmented(file_list, output_root, target_height, target_width, num_classes,
                                          batch_size, shuffle, augment, seed, augment_workers)

    rng = np.random.default_rng(seed)

    def generator():
        indices = np.arange(len(file_list))
        if shuffle:
            np.random.shuffle(indices)
        for index in indices:
            sample = load_sample(file_list[index], output_root, num_classes)
            yield augment_sample(*sample, rng, augment) if augment != "none" else sample

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(target_height, target_width, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(target_height, target_width, num_classes), dtype=tf.float32),
        ),
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _create_parallel_augmented(file_list, output_root, target_height, target_width, num_classes, batch_size,
                               shuffle, augment, seed, workers):
    """Same augmentation, run in `workers` parallel calls. Each sample gets its own seed drawn in order from
    default_rng(seed), and map(deterministic=True) keeps the order, so a run is reproducible. The random
    stream differs from the single-worker path, so results are not bit-identical to augment_workers=1."""
    seeds = np.random.default_rng(seed)

    def generator():
        indices = np.arange(len(file_list))
        if shuffle:
            np.random.shuffle(indices)
        for index in indices:
            yield int(index), int(seeds.integers(0, 2**63 - 1))

    def load_aug(index, sample_seed):
        image, mask = load_sample(file_list[int(index)], output_root, num_classes)
        return augment_sample(image, mask, np.random.default_rng(int(sample_seed)), augment)

    def tf_load(index, sample_seed):
        image, mask = tf.numpy_function(load_aug, [index, sample_seed], [tf.float32, tf.float32])
        image.set_shape((target_height, target_width, 1)); mask.set_shape((target_height, target_width, num_classes))
        return image, mask

    ds = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec((), tf.int64), tf.TensorSpec((), tf.int64)))
    ds = ds.map(tf_load, num_parallel_calls=workers, deterministic=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
