from src.model.model import AlexNet
from src.utils.callbacks.checkpoint import TrimmedModelCheckpoint
import tensorflow as tf
import datetime

# Hyperparameters
batch_size = 128
root_folder = ""

tf.random.set_seed(42)

# Creating logs
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + current_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

save_best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

callbacks = [TrimmedModelCheckpoint(), tensorboard_callback, save_best_model_callback]

alexnet = AlexNet(input_shape=(256, 256), metrics=['accuracy'], callbacks=callbacks)
alexnet.train(batch_size=batch_size, epochs=90, dataset_path=root_folder, validation=True)

tf.print("Learning process is over!")