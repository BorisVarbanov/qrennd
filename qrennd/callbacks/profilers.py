import time

from tensorflow import keras


class EpochRuntime(keras.callbacks.Callback):
    def __init__(self):
        super(EpochRuntime, self).__init__()

    def on_train_begin(self, logs=None):
        self.durations = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time_end = time.time()
        duration = epoch_time_end - self.epoch_time_start
        self.durations.append(duration)
        print(f"\nEpoch {epoch + 1}: was executed in {duration:.7f} seconds")
