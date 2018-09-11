import pytest
import random

from ts import TS

cities = ['A', 'B', 'C', 'D']
distances = [
    [0, 1, 2, 3],
    [1, 0, 3, 4],
    [2, 3, 0, 5],
    [3, 4, 5, 0]
]


def test_length():
    ts = TravellingSalesman(cities, distances)
    tour = [0, 1, 2, 3]
    # 1 + 3 + 5 + 3 = 12
    assert ts._length(tour) == 12
    random.shuffle(tour)
    tour_rev = list(reversed(tour))
    assert ts._length(tour) == ts._length(tour_rev)


def test_exhaustive_search():
    ts = TravellingSalesman(cities, distances)
    d, t = ts.exhaustive_search()
    assert d == 12
    assert t == cities


def hill_climb():
    ts = TravellingSalesman(cities, distances)
    assert sorted(cities) == sorted(ts.hill_climb())
