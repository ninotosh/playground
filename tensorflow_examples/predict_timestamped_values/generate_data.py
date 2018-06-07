import datetime
import pandas
import random
from typing import Iterator, Tuple


def generate_timestamps(n: int) -> Iterator[Tuple[datetime.datetime, int]]:
    t = datetime.datetime(year=2018, month=1, day=1)
    for _ in range(n):
        # add 7 days at most
        t += datetime.timedelta(seconds=random.randint(0, 60*60*24*7))
        # yield t, 10 if t.weekday() in [5, 6] else 0  # positive for weekends
        # yield t, 10 if t.day <= 10 else 0  # positive for early days in every month
        yield t, 10 if t.month in [1, 2, 12] else 0  # positive for winter
        # yield t, 3 if t.year >= 2019 else 0  # positive for year 2019 or later


if __name__ == '__main__':
    for data_set, n_examples in [('training', 400), ('validation', 150), ('prediction', 150)]:
        data_frame = pandas.DataFrame(generate_timestamps(n_examples), columns=['acted_at', 'value'])
        data_frame.to_csv('dummy_{}.csv'.format(data_set), index=False)
