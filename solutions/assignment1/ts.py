import itertools as it
import math
import random
import statistics

from selection_operators import (
    fitness_proportional,
    stochastic_universal_sampling,
    uniform
)
from crossover_operators import (
    edge_recombination,
    partially_mapped_crossover
)
from mutation_operators import (
    scramble_mutation,
    swap_mutation
)
from replacement_operators import (
    fitness
)


class TS(object):

    default_explore_exploite_ratio = 0.7
    default_keep_minimum = 5
    default_keep_ratio = 0.05
    default_hill_climb_max_limit = 1000

    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances

    def exhaustive_search(self):
        # Since it does not matter
        # 1) where in a cycle we start, and
        # 2) which direction we go through the cycle
        # the number of unique cycles is (n - 1)!/2.
        n = math.factorial(len(self.cities) - 1) // 2
        cycles = it.permutations(list(range(len(self.cities))))
        c = list(next(cycles))
        shortest = self._length(c), c
        for _ in range(n - 1):
            c = list(next(cycles))
            if self._length(c) < shortest[0]:
                shortest = self._length(c), c
        return shortest

    def hill_climb(self, op=swap_mutation, tour=None, limit=None):
        if not tour:
            tour = self._random_tour()
        max_limit = min(math.factorial(self._size - 1) // 2,
                        self.default_hill_climb_max_limit)
        limit = min(limit, max_limit) if limit else max_limit
        best = tour

        for _ in range(limit):
            other = op(tour[1])
            if self._length(other) < best[0]:
                best = self._length(other), other

        if best < tour:
            return self.hill_climb(tour=best, limit=limit)

        return best

    def genetic(self, pop_size, data=None, params=None):
        if not params:
            params = dict()
        params['pop_size'] = pop_size
        params = self._init_genetic(params)
        return self._search(params, data)

    def hybrid(self, kind, pop_size, data=None, params=None):
        if not params:
            params = dict()
        params['pop_size'] = pop_size
        if kind.startswith(('B', 'b')):
            params = self._init_hybrid_baldwinian(params)
        else:
            params = self._init_hybrid_lamarickan(params)
        return self._search(params, data)

    @property
    def _size(self):
        return len(self.cities)

    def _length(self, cycle):
        vs = cycle + [cycle[0]]
        dists = [self.distances[vs[i]][vs[i+1]]
                 for i in range(len(cycle))]
        return sum(dists)

    def _random_tour(self):
        cycle = list(range(len(self.cities)))
        random.shuffle(cycle)
        return self._length(cycle), cycle

    def _named_tour(self, tour):
        return tour[0], [self.cities[i] for i in tour[1]]

    def _stats(self, tours):
        # lengths = [self._length(t[1]) for t in tours]
        lengths = [t[0] for t in tours]
        best = min(lengths)
        mean = statistics.mean(lengths)
        worst = max(lengths)
        stdev = statistics.stdev(lengths)
        return best, mean, worst, stdev

    def _create_offsprint(self, parents, crossover_op, mutation_op, how_many):
        offspring = list()
        for _ in range(how_many):
            fp, sp = uniform(parents, 2)
            child = mutation_op(crossover_op(fp[1], sp[1]))
            offspring.append((self._length(child), child))
        return offspring

    def _search(self, params, data):

        def aux(params):
            # operators
            select = params['selection_op']
            cross = params['crossover_op']
            mutate = params['mutation_op']
            replace = params['replacement_op']

            pop = params['pop']
            pop_size = len(pop)

            # select parents
            parents = select(pop, len(pop) // 2)
            # create offspring
            children = self._create_offsprint(parents, cross, mutate, pop_size)
            # replacement
            keep = max(int(params['keep_ratio'] * pop_size),
                       params['keep_minimum'])
            pop = replace(pop, children, keep)
            params['pop'] = pop
            # add stats
            params['stats'].append(self._stats(pop))
            params['limit'] = params['limit'] - 1

            if params['limit'] < 1:
                return params
            return aux(params)

        def explore(params):
            params['selection_op'] = stochastic_universal_sampling
            params['crossover_op'] = partially_mapped_crossover
            params['mutation_op'] = scramble_mutation
            params['replacement_op'] = fitness
            return aux(params)

        def exploite(params):
            params['selection_op'] = fitness_proportional
            params['crossover_op'] = edge_recombination
            params['mutation_op'] = swap_mutation
            params['replacement_op'] = fitness
            return aux(params)

        params['limit'] = params['explore_limit']
        params = explore(params)

        params['limit'] = params['exploite_limit']
        params = exploite(params)

        if data is not None:
            data.clear()
            data.extend(params['stats'])
        return min(params['pop'])

    def _init_genetic(self, params):
        pop_size = params['pop_size']
        pop = [self._random_tour() for _ in range(pop_size)]
        stats = self._stats(pop)

        if 'explore_exploite_ratio' not in params:
            params['explore_exploite_ratio'] = (
                self.default_explore_exploite_ratio)

        params['explore_limit'] = (
            int(params['limit'] * params['explore_exploite_ratio']))
        params['exploite_limit'] = (
            params['limit'] - params['explore_limit'])
        params['pop'] = (
            pop)
        params['stats'] = (
            [stats])
        params['keep_minimum'] = (
            params.get('keep_minimum', self.default_keep_minimum))
        params['keep_ratio'] = (
            params.get('keep_ratio', self.default_keep_ratio))

        return params

    def _init_hybrid_lamarickan(self, params):
        params = self._init_genetic(params)
        pop = params['pop']
        pop = [self.hill_climb(tour=t, limit=100) for t in pop]
        params['pop'] = pop
        params['stats'] = [self._stats(pop)]

        return params

    def _init_hybrid_baldwinian(self, params):
        params = self._init_genetic(params)
        pop = params['pop']
        pop = [(self.hill_climb(tour=t, limit=100)[0], t[1]) for t in pop]
        pop_size = len(pop)
        parents = fitness_proportional(pop, pop_size // 2)
        cross = edge_recombination
        mutate = swap_mutation
        children = self._create_offsprint(parents, cross, mutate, pop_size)
        keep = max(int(params['keep_ratio'] * pop_size),
                   params['keep_minimum'])
        result = sorted(pop)[:keep]
        result = [(self._length(t[1]), t[1]) for t in result]
        how_many = pop_size - keep
        result.extend(sorted(children)[:how_many])
        params['pop'] = result
        params['stats'].append(self._stats(result))
        params['limit'] = params['limit'] - 1

        return params
