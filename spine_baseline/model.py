from tensorflow import keras

# Some TensorFlow distributions expose Keras through the ``tf.keras`` lazy
# attribute without installing an importable ``tensorflow.keras`` module.
# Importing the public ``keras`` attribute keeps the model definition working
# in both layouts while preserving the same Keras classes and behavior.
Model = keras.Model
layers = keras.layers


PAPER_ENCODER_FILTERS = (16, 32, 64, 128, 256)
PAPER_BOTTLENECK_FILTERS = 512
PAPER_DECODER_FILTERS = (256, 128, 64, 32)
PAPER_DROPOUTS = {
    "c1": 0.1,
    "c2": 0.1,
    "c3": 0.2,
    "c4": 0.2,
    "c5": 0.3,
    "bottleneck": 0.3,
    "u1": 0.2,
    "u2": 0.2,
    "u3": 0.1,
    "u4": 0.1,
}


def _dropout_rate(name: str, override: float | None) -> float:
    if override is not None:
        return override
    return PAPER_DROPOUTS[name]


def conv_bn_dropout(x, filters: int, dropout_rate: float):
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_uniform",
    )(x)
    x = layers.BatchNormalization()(x)
    return layers.Dropout(dropout_rate)(x)


def conv_only(x, filters: int):
    return layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_uniform",
    )(x)


def paper_conv_block(x, filters: int, dropout_rate: float):
    x = conv_bn_dropout(x, filters, dropout_rate)
    return conv_only(x, filters)


def upsample_block(x, skip, filters: int, kernel_size: int, leaky_relu_alpha: float):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=kernel_size,
        strides=2,
        padding="same",
        kernel_initializer="glorot_uniform",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    return layers.Concatenate()([x, skip])


def build_modified_unet(
    input_shape=(512, 640, 1),
    num_classes: int = 4,
    dropout_rate: float | None = None,
    leaky_relu_alpha: float = 0.1,
) -> Model:
    """Build the Ahmed et al. paper-aligned Modified U-Net baseline.

    The filter widths and dropout schedule follow the architecture figure and
    layer listing in the paper: 16, 32, 64, 128, 256 encoder filters plus an
    additional 512-channel bottleneck layer, then 256, 128, 64, 32 decoder
    filters. Passing dropout_rate overrides the paper schedule for ablations.
    """
    inputs = layers.Input(shape=input_shape)

    c1 = paper_conv_block(inputs, PAPER_ENCODER_FILTERS[0], _dropout_rate("c1", dropout_rate))
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = paper_conv_block(p1, PAPER_ENCODER_FILTERS[1], _dropout_rate("c2", dropout_rate))
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = paper_conv_block(p2, PAPER_ENCODER_FILTERS[2], _dropout_rate("c3", dropout_rate))
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = paper_conv_block(p3, PAPER_ENCODER_FILTERS[3], _dropout_rate("c4", dropout_rate))
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = paper_conv_block(p4, PAPER_ENCODER_FILTERS[4], _dropout_rate("c5", dropout_rate))
    bottleneck = paper_conv_block(c5, PAPER_BOTTLENECK_FILTERS, _dropout_rate("bottleneck", dropout_rate))

    u1 = upsample_block(bottleneck, c4, PAPER_DECODER_FILTERS[0], kernel_size=3, leaky_relu_alpha=leaky_relu_alpha)
    u1 = paper_conv_block(u1, PAPER_DECODER_FILTERS[0], _dropout_rate("u1", dropout_rate))
    u2 = upsample_block(u1, c3, PAPER_DECODER_FILTERS[1], kernel_size=2, leaky_relu_alpha=leaky_relu_alpha)
    u2 = paper_conv_block(u2, PAPER_DECODER_FILTERS[1], _dropout_rate("u2", dropout_rate))
    u3 = upsample_block(u2, c2, PAPER_DECODER_FILTERS[2], kernel_size=2, leaky_relu_alpha=leaky_relu_alpha)
    u3 = paper_conv_block(u3, PAPER_DECODER_FILTERS[2], _dropout_rate("u3", dropout_rate))
    u4 = upsample_block(u3, c1, PAPER_DECODER_FILTERS[3], kernel_size=2, leaky_relu_alpha=leaky_relu_alpha)
    u4 = paper_conv_block(u4, PAPER_DECODER_FILTERS[3], _dropout_rate("u4", dropout_rate))

    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation="softmax",
        kernel_initializer="glorot_uniform",
    )(u4)
    return Model(inputs, outputs, name="Ahmed2025_Modified_UNet")
