import scipy.spatial.distance as ssd
import numpy as np
import mappertools.mapper.clustering as mclust

from contextlib import contextmanager
import time

# https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
@contextmanager
def elapsed_timer():
    start = time.process_time()
    elapser = lambda: time.process_time() - start
    yield lambda: elapser()
    end = time.process_time()
    elapser = lambda: end-start


def form_data(N):
    X = np.random.rand(N,3) + np.array([[10,10,10]])
    Y = np.random.rand(N,3) + np.array([[-10,-10,-10]])
    data = np.concatenate((X,Y),axis=0)

    return data


def do_timings(metric, data):
    with elapsed_timer() as elapsed:
        k_fly = mclust.kMedoids(metric=metric, heuristic=2).fit(data)
        print("on-the-fly  {} kMedoids in {:.6f} seconds".format(metric, elapsed()))

    with elapsed_timer() as elapsed:
        distance_matrix = ssd.squareform(ssd.pdist(data, metric=metric))
        k_pre = mclust.kMedoids(metric="precomputed", heuristic=2).fit(distance_matrix)
        print("precomputed {} kMedoids in {:.6f} seconds".format(metric, elapsed()))


for n in range(100,200,100):
    data = form_data(n)
    do_timings("cosine", data)
    do_timings("euclidean", data)
    do_timings("chebyshev", data)
