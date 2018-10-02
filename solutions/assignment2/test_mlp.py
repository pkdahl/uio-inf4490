import random

from mlp import MLP


def test_init_weights():
    random.seed(7)
    expected = [[-0.35233447033367526, -0.6983016521509962],
                [0.3018689460797075, -0.8551274266649145],
                [0.0717640086133784, -0.2686221661748289],
                [-0.8840021504505864, 0.014871466378840514]]
    res = MLP._init_weights(3, 2)
    assert res == expected

    expected = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]
    res = MLP._init_weights(3, 3, "zeros")
    assert res == expected


def test_add():
    a = [[1,2],[3,4],[5,6]]
    b = [[2,4],[6,8],[10,12]]
    res = MLP._add(a,a)
    assert res == b


def test_scale():
    a = [[1,2],[5,6],[9,10]]
    scalar = 2
    expected = [[2,4],[10,12],[18,20]]
    res = MLP._scale(scalar, a)
    assert res == expected


def test_transpose():
    xs = [[1, 2, 3], [4, 5, 6]]
    res = MLP._transpose(xs)
    assert res == [[1,4], [2,5], [3,6]]


def test_fwd():
    x = [0, 1]
    ws = [[-1, 0], [0, 1], [1, 1]]
    u = [-1.0, 0.0]
    res = MLP._fwd(x, ws)
    assert res == u


def test_weights_update():
    print("-- Test weights update --")

    x = [1, 2, -1]
    d = [1, 2]
    update = [[1, 2], [2, 4], [-1, -2]]
    res = MLP._weights_update(x, d)
    print(res)
    assert res == update

    x = [0, 1, -1]
    d = [-1, 2]
    update = [[0, 0], [-1, 2], [1, -2]]
    res = MLP._weights_update(x, d)
    print(res)
    assert res == update


def test_deltas_hidden():
    print("-- Test deltas hidden --")

    hidden_out = [1, 2]
    deltas_out = [1, 2]
    weights_out = [[1, -1], [0, 1], [1, 1]]
    deltas_hidden = [-1, 2]
    res = MLP._deltas_hidden(lambda x: 1, hidden_out, weights_out, deltas_out)
    print(res)
    assert res == deltas_hidden


def test_train():

    inputs = [[0.0, 1.0]]
    targets = [[1.0, 0.0]]
    weights_in = [[-1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]]
    weights_in_update = [[0.0, 0.0],
                         [-1.0, 2.0],
                         [1.0, -2.0]]
    weights_in_after_update = [[-1.0, 0.0],
                               [0.1, 0.8],
                               [1.1, 0.8]]
    weights_hidden = [[1.0, -1.0],
                      [0.0, 1.0],
                      [1.0, 1.0]]
    weights_hidden_update = [[1.0, -2.0],
                             [-2.0, -4.0],
                             [1.0, 2.0]]
    weights_hidden_after_update = [[0.9, -1.2],
                                   [-0.2, 0.6],
                                   [0.9, 0.8]]

    mlp = MLP(inputs, targets, 2, eta=0.1)
    mlp.weights_in = weights_in
    mlp.weights_hidden = weights_hidden

    def g(x, beta=1.0):
        return x

    def dg(x, beta=1.0):
        return 1

    def add_bias(xs):
        return xs + [1]

    mlp._activation = g
    mlp._activation_derivative = dg
    MLP._add_bias = add_bias

    mlp.train(inputs, targets, iterations=1)
    assert mlp.weights_in == weights_in_after_update
    assert mlp.weights_hidden == weights_hidden_after_update


if __name__ == '__main__':
    test_add()
    test_scale()
    test_transpose()
    test_fwd()
    test_train()
    test_init_weights()
    # test_weights_update()
    # test_deltas_hidden()
