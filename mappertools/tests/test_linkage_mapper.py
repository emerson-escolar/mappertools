import pytest
import scipy.cluster.hierarchy
import scipy.spatial.distance as spd
import numpy as np

import mappertools.linkage_mapper as lk

def test_num_clusters():
    X = np.random.normal(size=(400,2))

    Z = scipy.cluster.hierarchy.linkage(X, method='single', metric='euclidean')
    merge_distances = Z[:,2]

    for i,t in enumerate(reversed(merge_distances)):
        labels = scipy.cluster.hierarchy.fcluster(Z, t, criterion='distance')

        assert len(set(labels)) == i+1
        assert t == lk.cluster_number_to_threshold(len(set(labels)), merge_distances)


def test_heuristics():
    # artificial data with "obvious" clustering
    X = np.array([[0,0,0],
                  [0,1,0],
                  [1,0,0],
                  [0,0,1],
                  [100,0,0],
                  [100,1,0],
                  [101,0,0],
                  [100,0,1],
                  [0,0,100],
                  [0,1,100],
                  [1,0,100],
                  [0,0,101]])

    fg = lk.LinkageMapper(heuristic='firstgap').fit(X)
    assert len(np.unique(fg.labels_)) == 3

    sil = lk.LinkageMapper(heuristic='sil').fit(X)
    assert len(np.unique(sil.labels_)) == 3


def test_heuristics_precomputed():
    # artificial data with "obvious" clustering
    X = np.array([[0,0,0],
                  [0,1,0],
                  [1,0,0],
                  [0,0,1],
                  [100,0,0],
                  [100,1,0],
                  [101,0,0],
                  [100,0,1],
                  [0,0,100],
                  [0,1,100],
                  [1,0,100],
                  [0,0,101]])
    dists = spd.pdist(X)

    fg = lk.LinkageMapper(heuristic='firstgap', metric='precomputed').fit(dists)
    assert len(np.unique(fg.labels_)) == 3

    sil = lk.LinkageMapper(heuristic='sil', metric='precomputed').fit(dists)
    assert len(np.unique(sil.labels_)) == 3
