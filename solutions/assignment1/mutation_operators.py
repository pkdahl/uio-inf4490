"""Mutation operators"""

import itertools as it
import random


def insert_mutation(genotype):
    """Picks two random alleles, move the second to follow the first
    shifiting the rest as necessary.
    """
    fst, snd = _random_pair(genotype)
    head = genotype[:fst+1]
    x = genotype[snd]
    tail = genotype[fst+1:]
    tail.remove(x)
    return head + [x] + tail


def inversion_mutation(genotype):
    """Pick a random subset of the genotype and reverse this subset."""
    fst, snd = _random_pair(genotype)
    head = genotype[:fst]
    mid = genotype[fst:snd+1]
    tail = genotype[snd+1:]
    mid.reverse()
    return head + mid + tail


def scramble_mutation(genotype):
    """Pick a random subset of the genotype and randomly reorder the alleles
    of this subset."""
    fst, snd = _random_pair(genotype)
    head = genotype[:fst]
    mid = genotype[fst:snd+1]
    tail = genotype[snd+1:]
    random.shuffle(mid)
    return head + mid + tail


def swap_mutation(genotype):
    """Pick two random alleles, swap their positions in the genotype."""
    fst, snd = _random_pair(genotype)
    result = genotype[:]
    result[fst] = genotype[snd]
    result[snd] = genotype[fst]
    return result


# -- Helper functions --------------------------------------------------------

def _random_pair(genotype):
    n = len(genotype)
    return random.choice(list(it.combinations(range(n), 2)))
