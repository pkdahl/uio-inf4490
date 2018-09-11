import pytest

from crossover_operators import order_crossover, edge_recombination

fp = [1, 2, 3, 4, 5, 6, 7, 8, 9]
sp = [9, 3, 7, 8, 2, 6, 5, 1, 4]


def test_order():
    child = order_crossover(fp, sp, False)
    assert child == [3, 8, 2, 4, 5, 6, 7, 1, 9]


def test_edge_recombination():
    child = edge_recombination(fp, sp)
    assert child == [1, 5, 6, 2, 8, 7, 3, 9, 4]
