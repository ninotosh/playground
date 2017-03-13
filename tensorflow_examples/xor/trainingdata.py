import numpy as np


class TrainingData(object):
    def __init__(self, inputs, targets):
        self.epoch = 0
        self.index = 0
        self.data = np.array(list(zip(inputs, targets)))
        np.random.shuffle(self.data)

    def next_batch(self, size=None):
        if size is None:
            size = len(self.data)

        end = self.index + size
        if end > len(self.data):
            return np.array([]), np.array([])

        batch = self.data[self.index:end]
        self.index = end
        # split it to (inputs, targets)
        return np.array([row[0] for row in batch]), np.array([row[1] for row in batch])

    def renew_epoch(self):
        self.epoch += 1
        self.index = 0
        np.random.shuffle(self.data)
