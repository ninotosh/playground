import unittest

from tensor_flow.autoencoder.trainingdata import TrainingData


class Test(unittest.TestCase):
    def test_training_data(self):
        data = TrainingData(4, 2)
        for d in data:
            self.assertEqual(len(d), 2)
            self.assertEqual(len(d[0]), 4)

        data.renew_epoch()
        for d in data:
            self.assertEqual(len(d), 2)
            self.assertEqual(len(d[0]), 4)
