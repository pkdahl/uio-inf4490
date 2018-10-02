"""Multi-level Preceptron with one hidden layer"""

import functools
import itertools
import math
import operator
import random


class MLP(object):

    def __init__(self, inputs, targets, n_hidden,
                 beta=1.0, eta=0.1, momentum=0.0):
        self.beta = beta
        self.eta = eta
        self.momentum = momentum
        self.n_in = self._get_n(inputs)
        self.n_hidden = n_hidden
        self.n_out = self._get_n(targets)
        self.n_data = len(inputs)
        self.weights_in = self._init_weights(self.n_in, self.n_hidden, self.n_data)
        self.weights_hidden = self._init_weights(self.n_hidden, self.n_out, self.n_data)

    def earlystopping(self, inputs, targets, valid_inputs, valid_targets,
                      iterations=100, stop_pct=0.1, verbose=False):
        msg = "Not the same number of inputs and targets"
        assert len(inputs) == len(targets), msg
        msg = "Validation inputs and validation targets have different cardinality"
        assert len(valid_inputs) == len(valid_targets), msg

        best = self.weights_in, self.weights_hidden
        smallest_error = None
        go = True
        epochs = 0
        limit = None
        while go:
            self.train(inputs, targets, iterations=iterations)
            epochs += 1
            outputs = [self.forward(x) for x in valid_inputs]
            error = sum([self._error(y, t) for y, t in zip(outputs, valid_targets)])
            error = error / len(outputs)

            improve_pct = None
            if smallest_error:
                improve_pct = (smallest_error - error) / smallest_error * 100.0

            if improve_pct and improve_pct < stop_pct:
                go = False
            else:
                if limit:
                    limit = None
                    print("removed limit")
                smallest_error = error
                best = self.weights_in, self.weights_hidden

            if verbose:
                msg = "-- "
                msg += f"epochs: {epochs:>3} -- "
                msg += f"error: {error:2.5f} -- "
                if not improve_pct:
                    msg += "improvement: N/A     --"
                else:
                    msg += f"improvement: {improve_pct:2.5f} --"
                print(msg)

        self.weights_in, self.weights_hidden = best

    def train(self, inputs, targets, iterations=100):
        n_data = float(len(inputs))
        prev_updates = self._init_updates()

        for _ in range(iterations):
            updates = self._init_updates()
            for inp, target in zip(inputs, targets):

                hidden = self._fwd(inp, self.weights_in)
                hidden = [self._activation(x, beta=self.beta) for x in hidden]
                output = self._fwd(hidden, self.weights_hidden)
                deltas = self._calc_deltas(target, output, hidden)
                updates = self._adjust_updates(updates, deltas, inp, hidden)

            prev_updates = self._update_weights(n_data, updates, prev_updates)

    def forward(self, inp):
        hidden_in = self._fwd(inp, self.weights_in)
        hidden_out = [self._activation(x, beta=self.beta) for x in hidden_in]

        return self._fwd(hidden_out, self.weights_hidden)

    def confusion(self, inputs, targets, show=True):
        # actual class along columns
        # predicted class along rows
        n_data = len(targets)
        confusion = [[0 for _ in range(self.n_out)] for _ in range(self.n_out)]

        results = [self.forward(x) for x in inputs]
        predicted = [r.index(max(r)) for r in results]
        actual = [t.index(max(t)) for t in targets]
        for a, p in zip(actual, predicted):
            confusion[p][a] += 1
        n_correct = sum([confusion[i][i] for i in range(self.n_out)])
        correct_pct = n_correct * 100.0 / n_data

        # Not exactly beautiful
        row_label_width = len(str(self.n_out))
        col_width = max(row_label_width,
                        max([len(str(i)) for i in itertools.chain(*confusion)]))
        rows = [f"{c} | " + " ".join([f"{i:>{col_width}}" for i in row])
                for c, row in enumerate(confusion)]
        row_width = len(rows[0])
        header = (" " * row_label_width +
                  " | " +
                  " ".join([f"{c:>{col_width}}"
                            for c, _ in enumerate(confusion)]))
        hline = ("-" * row_label_width +
                 "-+" +
                 "-" * (row_width - row_label_width - 2))
        rows = "\n".join(rows)

        if show:
            print()
            print(header)
            print(hline)
            print(rows)
            print(f"{correct_pct:.2f}% correctly classified")

        return n_correct, n_data

    def _init_updates(self):
        weights_in_hid = self._init_weights(self.n_in, self.n_hidden, kind='zeros')
        weights_hid_out = self._init_weights(self.n_hidden, self.n_out, kind='zeros')
        return weights_in_hid, weights_hid_out

    # def _update_deltas(self, target, output, hidden, deltas, derivative=None):
    #     if not derivative:
    #         derivative = functools.partial(self._activation_derivative, beta=self.beta)

    #     out_update = MLP._deltas_out(output, target)
    #     hid_update = MLP._deltas_hidden(derivative, hidden, self.weights_hidden, out_update)

    #     deltas_hid = [d + u for d, u in zip(deltas[0], hid_update)]
    #     deltas_out = [d + u for d, u in zip(deltas[1], out_update)]

    #     return deltas_hid, deltas_out

    def _calc_deltas(self, target, output, hidden, derivative=None):
        if not derivative:
            derivative = functools.partial(self._activation_derivative, beta=self.beta)

        deltas_out = self._deltas_out(output, target)
        deltas_hid = self._deltas_hidden(derivative, hidden, self.weights_hidden, deltas_out)

        return deltas_hid, deltas_out

    def _adjust_updates(self, updates, deltas, inp, hidden):
        in_hid = self._add(
            self._calc_weights_update(inp, deltas[0], eta=self.eta),
            updates[0])
        hid_out = self._add(
            self._calc_weights_update(hidden, deltas[1], eta=self.eta),
            updates[1])

        return in_hid, hid_out

    def _update_weights(self, n_data, updates, prev_updates):
        prev_in_hid = self._scale(self.momentum, prev_updates[0])
        prev_hid_out = self._scale(self.momentum, prev_updates[1])

        in_hid = self._scale(1/n_data, updates[0])
        hid_out = self._scale(1/n_data, updates[1])
        return_value = in_hid, hid_out

        in_hid = self._add(in_hid, prev_in_hid)
        hid_out = self._add(hid_out, prev_hid_out)

        self.weights_in = self._diff(self.weights_in, in_hid)
        self.weights_hidden = self._diff(self.weights_hidden, hid_out)

        return return_value

    @staticmethod
    def _get_n(lists):
        assert len(lists) > 0
        n = len(lists[0])
        assert all([True if len(l) == n else False
                    for l in lists])
        return n

    # @staticmethoda
    # def _scale_deltas(deltas, n_data):
    #     n = float(n_data)
    #     deltas_hid, deltas_out = deltas
    #     deltas_hid = [d / n for d in deltas_hid]
    #     deltas_out = [d / n for d in deltas_out]
    #     return deltas_hid, deltas_out

    @staticmethod
    def _fwd(inp, weights):
        xs = MLP._add_bias(inp)
        weights = MLP._transpose(weights)
        return [sum([x * w for x, w in zip(xs, ws)])
                for ws in weights]

    @staticmethod
    def _error(output, target):
        """Sum of squares error function.

        Equation 4.1 in the book.
        """
        assert len(output) == len(target)
        return 0.5 * sum([(y - t)**2 for y, t in zip(output, target)])

    @staticmethod
    def _activation(x, beta=1.0):
        return 1.0 / (1.0 + math.exp(-1.0 * beta * x))

    @staticmethod
    def _activation_derivative(x, beta=1.0):
        return beta * x * (1.0 - x)

    @staticmethod
    def _init_weights(n_in, n_out, n_data=1, kind="random"):
        if kind == "zeros":
            def zeros(): return 0.0
            w = zeros
        else:
            d = math.sqrt(n_data)
            def rand(): return random.uniform(-1.0/d, 1.0/d)
            w = rand

        return [[w() for _ in range(n_out)]
                for _ in range(n_in + 1)]

    @staticmethod
    def _add_bias(inp, bias=-1.0):
        return inp + [bias]

    @staticmethod
    def _calc_weights_update(xs, deltas, eta=1.0):
        xs = MLP._add_bias(xs)
        return [[eta * x * d for d in deltas]
                for x in xs]

    @staticmethod
    def _deltas_out(output, target):
        """Function 4.14 in the book."""
        return [y - t for y, t in zip(output, target)]

    @staticmethod
    def _deltas_hidden(derivative, hidden_out, weights_out, deltas_out):
        deltas= [sum([d * w for d, w in zip(deltas_out, weights)])
                 for weights in weights_out[:-1]]

        return [derivative(a) * d for a, d in zip(hidden_out, deltas)]

    @staticmethod
    def _add(left, right):
        return [[l + r for l, r in zip(ls, rs)]
                for ls, rs in zip(left, right)]

    @staticmethod
    def _diff(minuend, subtrahend):
        return [[m - s for m, s in zip(ms, ss)]
                for ms, ss in zip(minuend, subtrahend)]

    @staticmethod
    def _scale(scalar, mat):
        return [[scalar * e for e in row]
                for row in mat]

    @staticmethod
    def _transpose(list_2d):
        return [list(t) for t in zip(*list_2d)]
