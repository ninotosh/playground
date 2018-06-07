import collections
import functools
import math
from typing import Callable, Dict, Generator, Iterator, List, Tuple, Union  # NOQA

import pandas
import pandas.core.series
import pandas.io.parsers
import tensorflow as tf


BATCH_SIZE = 64
REPEAT = 1000
TRAINING_STEPS = 1000


TimestampField = collections.namedtuple('TimestampField', ['name', 'tf_type'])
TIMESTAMP_FIELDS = [
    TimestampField('year', tf.float16),
    TimestampField('month', tf.float16),
    TimestampField('day', tf.float16),
    TimestampField('days_in_month', tf.float16),
    TimestampField('dayofweek', tf.float16)
]
Timestamp = collections.namedtuple(
    'Timestamp',
    [field.name for field in TIMESTAMP_FIELDS]
)


def split_timestamp(timestamp: pandas.Timestamp, field_names) -> Tuple:
    return tuple(getattr(timestamp, name) for name in field_names)


def read_csv(file_path: str) -> Iterator[Tuple]:
    reader = pandas.read_csv(
        file_path,
        parse_dates=['acted_at'],
        iterator=True,
        chunksize=4
    )  # type: pandas.io.parsers.TextFileReader
    for data_frame in reader:  # type: pandas.DataFrame
        for _, series in data_frame.iterrows():  # type: Tuple[int, pandas.core.series.Series]
            acted_at = series['acted_at']  # type: pandas.Timestamp
            yield (
                *split_timestamp(acted_at, Timestamp._fields),
                series['value']
            )


def coordinate_on_circumference(
        value: tf.Tensor,
        upper: Union[int, float],
        trigonometric: Callable[[tf.Tensor], tf.Tensor]
) -> tf.Tensor:
    radian = 2 * math.pi * value / tf.constant(upper, dtype=tf.float16)
    return trigonometric(radian)


x_on_circumference = functools.partial(coordinate_on_circumference, trigonometric=tf.cos)
y_on_circumference = functools.partial(coordinate_on_circumference, trigonometric=tf.sin)


def get_data_set(generator: Generator) -> tf.data.Dataset:
    data_set = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(
            *(field.tf_type for field in TIMESTAMP_FIELDS),
            tf.int32
        ),
        output_shapes=(
            *tuple([()] * len(TIMESTAMP_FIELDS)),
            ()
        )
    )
    # data_set = data_set.skip(10000)
    data_set = data_set.batch(BATCH_SIZE)
    # data_set = data_set.take(1)
    data_set = data_set.shuffle(8192)
    data_set = data_set.repeat(REPEAT)
    return data_set


def input_fn_from_generator(generator: Generator[Tuple, None, None]):
    data_set = get_data_set(generator)
    iterator = data_set.make_one_shot_iterator()  # type: Iterator[Tuple]
    *feature_tensors, label_tensor = iterator.get_next()  # Tuple[List[tf.Tensor], tf.Tensor]
    timestamp = Timestamp._make(feature_tensors)

    if tf.executing_eagerly():
        for tensor in timestamp:
            assert tensor.shape.as_list() == [BATCH_SIZE]
        assert label_tensor.shape.as_list() == [BATCH_SIZE]

    features = {
        column.key: column.timestamp_field_fn(timestamp)
        for column in TIMESTAMP_FEATURE_COLUMNS
    }
    return features, label_tensor


TimestampFeatureColumn = collections.namedtuple(
    'TimestampFeatureColumn',
    ['key', 'normalizer_fn', 'timestamp_field_fn']
)
TIMESTAMP_FEATURE_COLUMNS = [
    TimestampFeatureColumn(
        key='year',
        normalizer_fn=(lambda year: (year - 2018.) / 10.),
        # normalizer_fn=(lambda year: year),
        timestamp_field_fn=(lambda t: t.year)
    ),
    TimestampFeatureColumn(
        key='month_x',
        normalizer_fn=(lambda month: x_on_circumference(month - 1, 12)),
        timestamp_field_fn=(lambda t: t.month)
    ),
    TimestampFeatureColumn(
        key='month_y',
        normalizer_fn=(lambda month: y_on_circumference(month - 1, 12)),
        timestamp_field_fn=(lambda t: t.month)
    ),
    TimestampFeatureColumn(
        key='day_x',
        normalizer_fn=(lambda day_ratio: x_on_circumference(day_ratio, 1)),
        timestamp_field_fn=(lambda t: (t.day - 1) / t.days_in_month)
    ),
    TimestampFeatureColumn(
        key='day_y',
        normalizer_fn=(lambda day_ratio: y_on_circumference(day_ratio, 1)),
        timestamp_field_fn=(lambda t: (t.day - 1) / t.days_in_month)
    ),
    TimestampFeatureColumn(
        key='dayofweek_x',
        normalizer_fn=(lambda dayofweek: x_on_circumference(dayofweek, 7)),
        timestamp_field_fn=(lambda t: t.dayofweek)
    ),
    TimestampFeatureColumn(
        key='dayofweek_y',
        normalizer_fn=(lambda dayofweek: y_on_circumference(dayofweek, 7)),
        timestamp_field_fn=(lambda t: t.dayofweek)
    )
]


def main(_: List[str]) -> None:
    """
    $ docker run -it --rm -p 6006:6006 \
        -v `pwd`/model_dir:/mnt \
        tensorflow/tensorflow:latest-py3 \
        tensorboard --logdir /mnt
    """
    # tf.enable_eager_execution()

    training_csv_path = 'dummy_training.csv'
    validation_csv_path = 'dummy_validation.csv'
    prediction_csv_path = 'dummy_prediction.csv'
    train = True
    validate = True
    predict = True

    if tf.executing_eagerly():
        # error: Estimators are not supported when eager execution is enabled.
        timestamp_batch_tensors = input_fn_from_generator(
            lambda: read_csv(training_csv_path)
        )  # type: Tuple[tf.Tensor]
        print(timestamp_batch_tensors)
    else:
        feature_columns = [
            tf.feature_column.numeric_column(
                key=column.key,
                normalizer_fn=column.normalizer_fn
            )
            for column in TIMESTAMP_FEATURE_COLUMNS
        ]
        regressor = tf.estimator.DNNRegressor(
            hidden_units=[16, 8, 4],
            feature_columns=feature_columns,
            config=tf.estimator.RunConfig(
                model_dir='/opt/project/model_dir',
                save_summary_steps=10,
                log_step_count_steps=10
            )
        )

        if train:
            regressor.train(
                input_fn=lambda: input_fn_from_generator(
                    lambda: read_csv(training_csv_path)
                ),
                steps=TRAINING_STEPS
            )

        if validate:
            result = regressor.evaluate(
                input_fn=lambda: input_fn_from_generator(
                    lambda: read_csv(validation_csv_path)
                )
            )  # type: Dict[str, float]

            print(result)

        if predict:
            read_csv_prediction = (lambda: read_csv(prediction_csv_path))
            prediction_batches = regressor.predict(
                input_fn=lambda: input_fn_from_generator(read_csv_prediction),
                yield_single_examples=False
            )  # type: Iterator[Dict[str, numpy.ndarray]]

            predictions = []  # type: List[float]
            for prediction_batch in prediction_batches:
                predictions += prediction_batch['predictions'].flatten().tolist()

            for time_fields, prediction in zip(read_csv_prediction(), predictions):
                print(time_fields, prediction)


if __name__ == '__main__':
    tf.app.run()
