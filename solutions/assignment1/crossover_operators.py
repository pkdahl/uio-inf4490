"""Crossover operators"""

import random
from collections import Counter


def edge_recombination(fst_parent, snd_parent):

    network = _create_network([fst_parent, snd_parent])
    child, network, links = _start_node(
        list(), network, fst_parent, snd_parent)
    while network:
        child, network, links = _next_node(child, network, links)
    return child


def partially_mapped_crossover(fst_parent, snd_parent, swap_parents=None):
    """Partially mapped crossover"""
    n = len(fst_parent)
    # assert len(snd_parent) == n

    fp, sp = _swap_parents(fst_parent, snd_parent, swap_parents)
    child = [None] * n
    st, en = sorted([random.randrange(n), random.randrange(n)])
    positions = [sp.index(x) for x in fp[st:en] if x not in sp[st:en]]
    elements = [x for x in sp[st:en] if x not in fp[st:en]]

    # move elements in segment of first parent to child
    for i in range(st, en):
        child[i] = fp[i]
    # move remaining elements in segment of second parent to child
    for i, el in zip(positions, elements):
        child[i] = el
    # fill empty positions in child with elements from same position
    # of second parent
    for i in range(n):
        if child[i] is None:
            child[i] = sp[i]

    return child


def order_crossover(fst_parent, snd_parent, swap_parents=None):
    n = len(fst_parent)
    fp, sp = _swap_parents(fst_parent, snd_parent, swap_parents)
    st, en = sorted([random.randrange(n), random.randrange(n)])
    en += 1
    child = [None] * n

    child[st:en] = fp[st:en]

    i, j = en, en
    while None in child:
        if sp[j % n] not in child:
            child[i % n] = sp[j % n]
            i += 1
        j += 1

    return child


def cycle_crossover(fst_parent, snd_parent, swap_parents=None):
    n = len(fst_parent)
    child = [None for _ in range(n)]
    fp, sp = _swap_parents(fst_parent[:], snd_parent[:], swap_parents)
    while None in child:
        cycle = _get_cycle(fp, sp)
        for i in cycle:
            child[i] = fp[i]
            fp[i] = None
            sp[i] = None
        fp, sp = sp, fp
    return child


# -- Helper functions --------------------------------------------------------

def _create_network(parents):
    """Create adjaceny matrix"""
    n = len(parents[0])
    assert all([len(p) == n for p in parents])
    network = dict()
    for p in parents:
        p = [p[-1]] + p + [p[0]]
        for i in range(n):
            v = p[i+1]
            c = network.get(v, Counter())
            c.update([p[i], p[i+2]])
            network[v] = c
    return network


def _start_node(child, network, fp, sp):
    node = random.choice([fp[0], sp[0]])
    child.append(node)
    network, links = _remove_node(network, node)
    return child, network, links


def _next_node(child, network, links):
    node = _common_link(network, links)
    if not node:
        node = _fewest_links(network, links)
    if not node:
        node = random.choice(list(network.keys()))

    child.append(node)
    network, links = _remove_node(network, node)
    return child, network, links


def _common_link(network, links):
    if not links:
        return None
    candidates = [n for n, cnt in links.most_common() if cnt > 1]
    if not candidates:
        return None
    return random.choice(candidates)


def _fewest_links(network, links):
    if not links:
        return None
    counts = list()
    for n in links:
        counts.append(len(network[n]))
    fewest = min(counts)
    candidates = [n for n in links if len(network[n]) == fewest]
    return random.choice(candidates)


def _remove_node(network, node):
    for n in network.keys():
        links = network[n]
        del links[node]
    links = network[node]
    del network[node]

    return network, links


def _swap_parents(fst_parent, snd_parent, swap=None):
    if swap is None:
        swap = random.randrange(2)

    if swap:
        fp = snd_parent
        sp = fst_parent
    else:
        fp = fst_parent
        sp = snd_parent

    return fp, sp


def _get_cycle(fst_parent, snd_parent):
    indicies = list()
    cycle = list()
    i = _first_element(fst_parent)
    while i not in indicies:
        indicies.append(i)
        i = fst_parent.index(snd_parent[i])
    for i in indicies:
        cycle.append(i)
    return cycle


def _first_element(it):
    for i in range(len(it)):
        if it[i] is not None:
            return i
    return None


if __name__ == '__main__':
    fp = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sp = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    print(edge_recombination(fp, sp))
