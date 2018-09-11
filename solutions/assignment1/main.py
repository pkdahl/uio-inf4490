import functools as ft
import statistics
import time
import matplotlib.pyplot as plt
from data import Data
from ts import TS


def run_exhaustive_search():
    def aux(n_cities):
        data = Data(n_cities)
        ts = TS(data.cities, data.distances)
        t = time.process_time()
        result = ts.exhaustive_search()
        elapsed_time = time.process_time() - t
        return result[0], elapsed_time

    results = list()
    max_dt = 120
    for n in range(6, 25):
        length, dt = aux(n)
        results.append((length, dt))
        print("n_cities:", n, "length:", length, "elapsed time:", dt)
        if dt > max_dt:
            break
    return results


def run_hill_climb():

    def aux(n_cities, n_runs, times):
        data = Data(n_cities)
        ts = TS(data.cities, data.distances)
        tours = list()
        for _ in range(n_runs):
            start = time.process_time()
            t = ts.hill_climb()
            dt = time.process_time() - start
            tours.append(t)
            times.append(dt)
        return tours

    for n in [10, 24]:
        times = list()
        tours = aux(n, 20, times)
        title = f"20 runs of hill climb with {n} cities"
        show_stats(tours, running_time=statistics.mean(times), title=title)
        filename = f"hill_climber_{n}.png"
        plot_tours(tours, show_plot=False, filename=filename)


def run_genetic_10(pop_size, n_generations, times):
    ts_data = Data(10)
    ts = TS(ts_data.cities, ts_data.distances)
    results = _run(ts.genetic, pop_size, n_generations, times)

    show_stats(results['tours'], statistics.mean(results['times']))
    filename = f"genetic_10.png"
    plot_tours(results['tours'],
               show_plot=False,
               filename=filename,
               ymin=7000, ymax=8500)
    return results


def run_genetic_24(pop_size, n_generations, times,
                   show_plot=True, save_fig=False):
    ts_data = Data(24)
    ts = TS(ts_data.cities, ts_data.distances)
    results = _run(ts.genetic, pop_size, n_generations, times)

    stats_title = (f"{times} runs on genetic with 24 cities, "
                   f"a population size of {pop_size} "
                   f"and {n_generations} generations")
    if save_fig:
        filename = f"24_genetic_pop_{pop_size}_gens_{n_generations}.png"
    else:
        filename = ''

    show_stats(results['tours'],
               running_time=statistics.mean(results['times']),
               title=stats_title)
    plot_tours(results['tours'], show_plot=show_plot, filename=filename)

    return results


def run_hybrid_baldwinian(pop_size, n_generations, times,
                          show_plot=True, save_fig=False):
    ts_data = Data(24)
    ts = TS(ts_data.cities, ts_data.distances)
    method = ft.partial(ts.hybrid, "baldwinian")
    results = _run(method, pop_size, n_generations, times)

    stats_title = (f"{times} runs on hybrid baldwinian with 24 cities, "
                   f"a population size of {pop_size} "
                   f"and {n_generations} generations")
    if save_fig:
        filename = (f"24_hybrid_baldwinian_pop_{pop_size}_"
                    f"gens_{n_generations}.png")
    else:
        filename = ''

    show_stats(results['tours'],
               running_time=statistics.mean(results['times']),
               title=stats_title)
    plot_tours(results['tours'], show_plot=show_plot, filename=filename)

    return results


def run_hybrid_lamarckian(pop_size, n_generations, times,
                          show_plot=True, save_fig=False):
    ts_data = Data(24)
    ts = TS(ts_data.cities, ts_data.distances)
    method = ft.partial(ts.hybrid, "lamarckian")
    results = _run(method, pop_size, n_generations, times)

    stats_title = (f"{times} runs on hybrid lamarckian with 24 cities, "
                   f"a population size of {pop_size} "
                   f"and {n_generations} generations")
    if save_fig:
        filename = (f"24_hybrid_lamarckian_pop_{pop_size}_"
                    f"gens_{n_generations}.png")
    else:
        filename = ''

    show_stats(results['tours'],
               running_time=statistics.mean(results['times']),
               title=stats_title)
    plot_tours(results['tours'], show_plot=show_plot, filename=filename)

    return results


def run_multiple(func, pop_sizes, n_generations, times,
                 show_plot=False, save_fig=True):
    results = dict()
    for pop_size in pop_sizes:
        results[pop_size] = func(pop_size, n_generations, times,
                                 show_plot=show_plot, save_fig=save_fig)

    if save_fig:
        filename = f"{func.__name__}_{times}_runs_{n_generations}_gens.png"
    else:
        filename = ''

    plot_gen_avg_bests(results, show_plot=show_plot, filename=filename)


def _run(ts_method, pop_size, n_generations, times):
    results = {
        'tours': list(),
        'gen_bests': list(),
        'times': list()
    }

    for _ in range(times):
        stats = list()
        params = {
            'explore_exploite_ratio': 0.6,
            'limit': n_generations
        }
        t = time.process_time()
        res = ts_method(pop_size, stats, params)
        elapsed_time = time.process_time() - t
        results['tours'].append(res)
        results['gen_bests'].append([t[0] for t in stats])
        results['times'].append(elapsed_time)

    results['gen_avg_bests'] = (
        [statistics.mean(t) for t in zip(*results['gen_bests'])])
    del results['gen_bests']

    return results


def plot_tours(tours, show_plot=True, filename=None, ymin=12000, ymax=18000):

    lengths = [t[0] for t in tours]

    def color(length):
        best = min(lengths)
        worst = max(lengths)
        if length == best:
            return 'blue'
        if length == worst:
            return 'red'
        return 'gray'

    data = {
        'x': list(range(len(lengths))),
        'y': lengths,
        'c': [color(l) for l in lengths]
    }
    avg = statistics.mean(lengths)
    stdev = statistics.stdev(lengths)
    xmax = max(data['x'])
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.scatter('x', 'y', c='c', data=data)
    plt.hlines(avg, 0, xmax, linewidth=1.0)
    plt.hlines(avg+stdev, 0, xmax, 'gray', linestyles='dashed', linewidth=0.8)
    plt.hlines(avg-stdev, 0, xmax, 'gray', linestyles='dashed', linewidth=0.8)
    plt.xticks([])
    if filename:
        plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.clf()


def plot_gen_avg_bests(results, show_plot=True, filename=None):
    colors = ['blue', 'red', 'green']
    ymin = 12000
    ymax = 28000

    def plot(ys, color, label):
        return plt.plot(ys, color, label=label)

    plt.ylim(ymin=ymin, ymax=ymax)
    for pop_size, color in zip(results.keys(), colors):
        ys = results[pop_size]['gen_avg_bests']
        plot(ys, color, pop_size)
    plt.xticks([])
    plt.legend()
    if filename:
        plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.clf()


def _plot_gen_stats(gens):
    xs = list(range(len(gens)))
    bests = [x[0] for x in gens]
    avgs = [x[1] for x in gens]
    worsts = [x[2] for x in gens]
    avg_plus_stdev = [x[1] + x[3] for x in gens]
    avg_minus_stdev = [x[1] - x[3] for x in gens]
    plt.plot(xs, bests, 'blue')
    plt.plot(xs, avgs, 'gray')
    plt.plot(xs, worsts, 'red')
    plt.plot(xs, avg_plus_stdev, 'gray', linestyle='--', linewidth=0.8)
    plt.plot(xs, avg_minus_stdev, 'gray', linestyle='--', linewidth=0.8)
    plt.xticks([])
    plt.show()


def show_stats(tours, running_time=None, title=''):
    lengths = [t[0] for t in tours]
    best = min(lengths)
    avg = statistics.mean(lengths)
    worst = max(lengths)
    stdev = statistics.stdev(lengths)
    if not title:
        title = f"Statistics over {len(tours)} tours"
    print()
    if title:
        print('  ' + title)
        print('-' * (len(title) + 4))
    print(f"Best tour has length {best}")
    print(f"Worst tour has length {worst}")
    print(f"Average over all tours is {avg}")
    print(f"Standard deviation over all tours is {stdev}")
    if running_time:
        print(f"Average running time was {running_time} seconds")
    print()


if __name__ == '__main__':
    run_exhaustive_search()
    run_hill_climb()
    run_genetic_10(200, 500, 20)
    pop_sizes = [25, 50, 100]
    times = 20
    n_generations = 1000
    algos = [run_genetic_24, run_hybrid_baldwinian, run_hybrid_lamarckian]
    for algo in algos:
        run_multiple(algo, pop_sizes, n_generations, times,
                     show_plot=False, save_fig=True)
