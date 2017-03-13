import itertools

import numpy as np


class TrainingData(object):
    def __init__(self, input_dim, batch_size=None):
        self.epoch = 0
        self.index = 0
        self.batch_size = batch_size if batch_size else input_dim
        # self.data = np.identity(input_dim, np.int)
        self.data = [np.array(x) for x in itertools.product([0, 1], repeat=input_dim)]
        # e.g. input_dim == 3
        # array([[0, 1, 0],
        #        [1, 0, 0],
        #        [0, 0, 1]])
        np.random.shuffle(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration

        end = min(self.index + self.batch_size, len(self.data))

        batch = self.data[self.index:end]
        self.index = end

        return batch

    def renew_epoch(self):
        self.epoch += 1
        self.index = 0
        np.random.shuffle(self.data)
