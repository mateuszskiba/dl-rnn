from keras import callbacks
import json


class HistoryCheckpoint(callbacks.Callback):
    PARAMS = ['acc', 'loss', 'val_acc', 'val_loss']

    def __init__(self, file_path, period):
        self.file_path = file_path
        self.epoch = 1
        self.history = {param: [] for param in HistoryCheckpoint.PARAMS}
        self.period = period

    def on_epoch_end(self, batch, logs={}):
        for param in HistoryCheckpoint.PARAMS:
            self.history[param].append(logs.get(param))

        if not (self.epoch % self.period):
            name = self.file_path
            try:
                name = name.format(epoch=self.epoch)
            except KeyError:
                pass

            with open(name, 'w+') as file:
                file.write(json.dumps(self.history))
        self.epoch += 1

