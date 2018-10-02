import itertools

import movements


from mlp import MLP

def cross_validation(k, inputs, targets, n_hidden, eta=0.1,
                     iterations=100, stop_pct=1, verbose=False):
    """k-fold cross-validation"""
    assert len(inputs) == len(targets)
    folds = create_folds(k, inputs, targets)
    combos = combinations(k)
    results = list()
    for valid_id, test_id in combos:
        data = select_folds(folds, valid_id, test_id)
        net = MLP(data['inputs'], data['targets'], n_hidden, eta=eta)
        net.earlystopping(data['inputs'], data['targets'],
                          data['valid_inputs'], data['valid_targets'],
                          iterations=iterations, stop_pct=stop_pct,
                          verbose=verbose)
        n_correct, n_data = net.confusion(data['test_inputs'],
                                          data['test_targets'],
                                          show=False)
        results.append(n_correct / n_data * 100.0)

        print(valid_id, test_id, n_correct / n_data * 100.0)

    return results


def create_folds(n_folds, inputs, targets):
    width = len(inputs) // n_folds
    folds = [(inputs[i*width:(i+1)*width], targets[i*width:(i+1)*width]) for i in range(n_folds-1)]
    folds.append((inputs[(n_folds-1)*width:], targets[(n_folds-1)*width:]))
    return folds


def combinations(k):
    combos = list()
    for fst, snd in itertools.combinations(list(range(k)), 2):
        combos.append((fst, snd))
        combos.append((snd, fst))
    return sorted(combos)


def select_folds(folds, valid_id, test_id):
    fold_ids = list(range(len(folds)))
    fold_ids.remove(valid_id)
    fold_ids.remove(test_id)

    inputs = list()
    targets = list()
    for fold_id in fold_ids:
        inputs.extend(folds[fold_id][0])
        targets.extend(folds[fold_id][1])
    valid_inputs, valid_targets = folds[valid_id]
    test_inputs, test_targets = folds[test_id]
    data = {
        'inputs': inputs,
        'targets': targets,
        'valid_inputs': valid_inputs,
        'valid_targets': valid_targets,
        'test_inputs': test_inputs,
        'test_targets': test_targets
    }
    return data


if __name__ == '__main__':
    import statistics

    movements, targets = movements.data()
    movements = movements[::,0:40].tolist()
    targets = targets.tolist()
    results = cross_validation(10, movements, targets, 8, eta=0.2,
                               iterations=50, stop_pct=0.1, verbose=True)
    mean = statistics.mean(results)
    stdev = statistics.stdev(results)

    print(results)
    print(f"Mean: {mean}")
    print(f"Standard deviation: {stdev}")
