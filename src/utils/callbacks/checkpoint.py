import tensorflow as tf

class TrimmedModelCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.print('Saving model without dense layers...')
        trimmed_model = tf.keras.Sequential(self.model.layers[:-1])

        trimmed_model.save('trimmed-checkpoints/saved_model-%d-%.4f' % (epoch, logs['val_accuracy']))

        tf.print('Model have been succesfully saved!')