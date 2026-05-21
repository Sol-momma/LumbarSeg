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


def create_dataset(file_list: list[str], output_root: Path, target_height: int, target_width: int,
                   num_classes: int, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    if not file_list:
        raise ValueError("file_list is empty")

    def generator():
        indices = np.arange(len(file_list))
        if shuffle:
            np.random.shuffle(indices)
        for index in indices:
            yield load_sample(file_list[index], output_root, num_classes)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(target_height, target_width, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(target_height, target_width, num_classes), dtype=tf.float32),
        ),
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
