import tensorflow as tf
from src.detaset.dataset import DatasetLoader


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
        layers.Dense(2048),
        layers.Dropout(0.5),
        layers.Dense(2048),
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

    return tf.keras.models.Model(inputs=input, outputs=output)


class AlexNet():
    def __init__(self, input_shape, metrics, callbacks):
        self.model = instantiate_model(input_shape=input_shape)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=metrics
        )

        self.input_shape = input_shape
        self.callbacks = callbacks

    
    def __call__(self, input):
        return self.model(input)
    

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)


    def train(self, batch_size=128, epochs=90, dataset_path="", validation=False):
        train_dataset_loader = DatasetLoader(dataset_path, volume='train')
        train_dataset = train_dataset_loader(image_size=self.input_shape, shuffle=True, batch_size=batch_size)

        if validation:
            validation_dataset_loader = DatasetLoader(dataset_path, volume='val')
            validation_dataset = validation_dataset_loader(image_size=self.input_shape, max=1000, shuffle=False)
        else:
            validation_dataset=None

        return self.model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks,
            validation_data=validation_dataset
        )