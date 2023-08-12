from src.model.model import instantiate_model
from src.detaset.dataset import DatasetLoader
from src.utils.callbacks.checkpoint import TrimmedModelCheckpoint
import tensorflow as tf

# Hyperparameters
batch_size = 32

# Loading Datasets
train_dataset_loader = DatasetLoader('dataset_path', volume='train')
train_dataset = train_dataset_loader(image_size=(256, 256), shuffle=True, batch_size=batch_size)

validation_dataset_loader = DatasetLoader('dataset_path', volume='train')
validation_dataset = validation_dataset_loader(image_size=(256, 256), shuffle=False)

# Training model
alexnet = instantiate_model(input_shape=(256, 256, 3))
print(alexnet.summary())

alexnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

history = alexnet.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=25,
    callbacks=[TrimmedModelCheckpoint()],
    validation_data=validation_dataset
)

tf.print("Learning process is over!")
tf.print("Friting history to the history-0.txt file")
with open('checkpoints/history-0.txt', 'w') as f:
    f.write(history)