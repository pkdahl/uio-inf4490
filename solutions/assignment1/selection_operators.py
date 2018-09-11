import random
import itertools as it


def uniform(pop, how_many):
    """With replacement"""
    return [random.choice(pop) for _ in range(how_many)]


def fitness_proportional(pop, how_many):

    def choice(pop):
        r = random.random() * _total_fitness(pop)
        for x, acc in zip(pop, it.accumulate([1/x[0] for x in pop])):
            if acc > r:
                return x
        return pop[-1]  # shouldn't be reachable

    return [choice(pop) for _ in range(how_many)]


def stochastic_universal_sampling(pop, how_many):
    result = list()
    f = _total_fitness(pop)
    r = random.random() * f / how_many
    for x, acc in zip(pop, it.accumulate([1/x[0] for x in pop])):
        if acc > r:
            result.append(x)
            r += f / how_many
    return result


def _total_fitness(pop):
        return sum([1/x[0] for x in pop])


if __name__ == '__main__':
    pop = list(enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9], 1))
    print("pop", pop)
    print(uniform(pop, 3))
    print(fitness_prop(pop, 5))
    print(stochastic_universal_sampling(pop, 3))
