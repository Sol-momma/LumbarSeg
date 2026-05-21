from tensorflow.keras import Model, layers


def conv_block(x, filters: int, dropout_rate: float, leaky_relu_alpha: float):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def upsample_block(x, skip, filters: int, leaky_relu_alpha: float):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", kernel_initializer="glorot_uniform")(x)
    x = layers.LeakyReLU(alpha=leaky_relu_alpha)(x)
    return layers.Concatenate()([x, skip])


def build_modified_unet(input_shape=(512, 640, 1), num_classes: int = 4,
                        dropout_rate: float = 0.5, leaky_relu_alpha: float = 0.1) -> Model:
    inputs = layers.Input(shape=input_shape)

    e1 = conv_block(inputs, 64, dropout_rate, leaky_relu_alpha)
    p1 = layers.MaxPooling2D(2)(e1)
    e2 = conv_block(p1, 128, dropout_rate, leaky_relu_alpha)
    p2 = layers.MaxPooling2D(2)(e2)
    e3 = conv_block(p2, 256, dropout_rate, leaky_relu_alpha)
    p3 = layers.MaxPooling2D(2)(e3)
    e4 = conv_block(p3, 512, dropout_rate, leaky_relu_alpha)
    p4 = layers.MaxPooling2D(2)(e4)

    bottleneck = conv_block(p4, 512, dropout_rate, leaky_relu_alpha)

    d4 = upsample_block(bottleneck, e4, 512, leaky_relu_alpha)
    d4 = conv_block(d4, 256, dropout_rate, leaky_relu_alpha)
    d3 = upsample_block(d4, e3, 256, leaky_relu_alpha)
    d3 = conv_block(d3, 128, dropout_rate, leaky_relu_alpha)
    d2 = upsample_block(d3, e2, 128, leaky_relu_alpha)
    d2 = conv_block(d2, 64, dropout_rate, leaky_relu_alpha)
    d1 = upsample_block(d2, e1, 64, leaky_relu_alpha)
    d1 = conv_block(d1, 64, dropout_rate, leaky_relu_alpha)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax", kernel_initializer="glorot_uniform")(d1)
    return Model(inputs, outputs, name="Modified_UNet")
