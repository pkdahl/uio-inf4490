def fitness(population=None, offspring=None, keep=None):
    if not keep:
        keep = 0
    result = sorted(population)[:keep]
    how_many = len(population) - keep
    result.extend(sorted(offspring)[:how_many])
    return result


def baldwinian():
    pass


def lamarckian():
    pass
