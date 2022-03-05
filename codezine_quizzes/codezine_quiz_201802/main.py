#!/usr/bin/env python3
import functools
import logging
import math

logger = logging.getLogger(__name__)


def count(reprints, copies):
    assert reprints >= 0
    assert copies > 0
    units = int(copies / 1000)
    return _count(reprints, units, units)


@functools.lru_cache(maxsize=None)
def _count(reprints, units, first_units_upper_bound):
    logger.debug('count(%s, %s, %s)', reprints, units, first_units_upper_bound)

    if units == 0:
        logger.debug('return 0 (units == 0)')
        return 0

    if reprints == 0:
        end_result = 1 if units <= first_units_upper_bound else 0
        logger.debug('return %s (reprints == 0)', end_result)
        return end_result

    result = 0
    min_first_units = int(math.ceil(units / float(reprints + 1)))
    for first_units in range(min_first_units, first_units_upper_bound + 1):
        remaining_units = units - first_units
        if remaining_units > 0:
            result += _count(reprints - 1, remaining_units, first_units)

    logger.debug('return %s', result)
    return result


def main():
    logging.basicConfig(level=logging.INFO)
    patterns = count(10, 100000)
    logger.info('%s pattern(s)', patterns)


if __name__ == '__main__':
    main()
