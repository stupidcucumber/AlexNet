from src.model.model import instantiate_model
from src.detaset.dataset import DatasetLoader
from src.utils.callbacks.checkpoint import TrimmedModelCheckpoint
import tensorflow as tf

# Hyperparameters
batch_size = 200
root_folder = ""

tf.random.set_seed(42)

# Loading Datasets
train_dataset_loader = DatasetLoader(root_folder, volume='train')
train_dataset = train_dataset_loader(image_size=(256, 256), shuffle=True, batch_size=batch_size)

validation_dataset_loader = DatasetLoader(root_folder, volume='val')
validation_dataset = validation_dataset_loader(image_size=(256, 256), max=1000, shuffle=False)

# Training model
alexnet = instantiate_model(input_shape=(256, 256, 3))
print(alexnet.summary())

alexnet.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = alexnet.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=90,
    callbacks=[TrimmedModelCheckpoint()],
    validation_data=validation_dataset
)

tf.print("Learning process is over!")
tf.print("Writing history to the history-0.txt file")
with open('checkpoints/history-0.txt', 'w') as f:
    f.write(history)