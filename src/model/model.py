import tensorflow as tf

class AlexNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AlexNet, self).__init__(**kwargs)

def instantiate_model(input_shape: tuple=(256, 256)):
    layers = tf.keras.layers

    block_1 = tf.keras.models.Sequential([
        layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(2, 2)),
        layers.BatchNormalization(axis=3), # normilize along the whole image
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Activation('relu')
    ])

    block_2 = tf.keras.models.Sequential([
        layers.Conv2D(filters=256, kernel_size=(5, 5)),
        layers.BatchNormalization(axis=3),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Activation('relu')
    ])

    block_3 = tf.keras.models.Sequential([
        layers.Conv2D(filters=384, kernel_size=(3, 3)),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu')
    ])

    block_4 = tf.keras.models.Sequential([
        layers.Conv2D(filters=384, kernel_size=(3, 3)),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu')
    ])

    block_5 = tf.keras.models.Sequential([
        layers.Conv2D(filters=256, kernel_size=(3, 3)),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu')
    ])

    calculate_output = tf.keras.models.Sequential([
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(1024),
        layers.Dropout(0.5),
        layers.Dense(1024),
        layers.Dropout(0.5),
        layers.Dense(1024),
        layers.Dropout(0.5),
        layers.Dense(1000, activation="softmax")
    ])

    input = layers.Input(shape=input_shape)

    x = block_1(input)
    x = block_2(x)
    x = block_3(x)
    x = block_4(x)
    x = block_5(x)

    output = calculate_output(x)

    return AlexNet(inputs=input, outputs=output)