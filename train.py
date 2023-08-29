from src.model.model import AlexNet
from src.utils.callbacks.checkpoint import TrimmedModelCheckpoint
import tensorflow as tf
import datetime
import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-shape", action="store", type=int, help="The size of side of the image.")
parser.add_argument("-b", "--batch_size", action="store", default=128, type=int)
parser.add_argument("-e", "--epochs", action="store", default=90, type=int)
parser.add_argument("-p", "--path_data", action="store", type=str)
parser.add_argument("-v", "--validate", action="store_true")
parser.add_argument("-m", "--model-path", action="store", type=str, help="If specified along with -l flag, the training will be started from pretrained model on the path.") 

if __name__ == "__main__":
    args = vars(parser.parse_args())

    input_shape = args['input_shape'], args['input_shape'], 3
    batch_size = args['batch_size']
    epochs = args['epochs']
    model_path = args['model_path']
    data_path = args['path_data']
    validate = args['validate']

    # Creating logs
    tf.random.set_seed(42)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + current_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    save_best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoint",
        monitor="val_accuracy" if validate else "accuracy",
        mode="max",
        save_best_only=True
    )

    callbacks = [TrimmedModelCheckpoint(), tensorboard_callback, save_best_model_callback]

    # Start training
    alexnet = AlexNet(
        input_shape=input_shape,
        metrics=['accuracy'],
        callbacks=callbacks
    )

    if model_path is not None:
        alexnet.load_model(model_path=model_path)

    alexnet.train(
        batch_size=batch_size,
        epochs=epochs,
        dataset_path=data_path,
        validation=validate
    )