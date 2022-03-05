import unittest

import main


class TestCount(unittest.TestCase):
    def test_count(self):
        for reprints, units, expected in [
            (1, 10000, 5),
            (2, 10000, 8),
        ]:
            self.assertEqual(
                expected,
                main.count(reprints, units),
                '{}, {}'.format(reprints, units)
            )

    def test__count(self):
        for reprints, units, upper, expected in [
            (0, 1, 1, 1),
            (1, 1, 1, 0),
            (0, 2, 2, 1),
            (1, 4, 4, 2),
            (2, 6, 1, 0),
            (2, 6, 2, 1),
            (2, 6, 3, 2),
            (2, 6, 4, 3),
            (2, 6, 6, 3),
            (2, 10, 10, 8),
        ]:
            self.assertEqual(
                expected,
                main._count(reprints, units, upper),
                '{}, {}, {}'.format(reprints, units, upper)
            )
