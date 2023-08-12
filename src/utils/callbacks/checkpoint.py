import tensorflow as tf

class TrimmedModelCheckpoint(tf.keras.callbacks):
    def on_epoch_end(self, epoch, logs=None):
        tf.print('Saving model without dense layers...')
        trimmed_model = tf.keras.Sequential(self.model.layers[:-1])

        trimmed_model.save('checkpoints/saved_model-%d-%.4f' % (epoch, logs['accuracy']))

        tf.print('Model have been succesfully saved!')