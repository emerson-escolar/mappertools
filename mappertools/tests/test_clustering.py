import pytest
import scipy.cluster.hierarchy
import scipy.spatial.distance as spd
import numpy as np

import mappertools.mapper.clustering as mclust

def test_two_clustering():
    X = np.random.rand(100,3)
    Y = np.random.rand(100,3) + np.array([[10,5,1]])
    data = np.concatenate((X,Y),axis=0)
    labels = mclust.kMedoids(metric="euclidean", heuristic=2, prefix=None).fit(data).labels_
    for i in range(100):
        assert labels[i] == labels[0]
        assert labels[i+100] == labels[100]

def test_two_clustering_kmeans():
    X = np.random.rand(100,3)
    Y = np.random.rand(100,3) + np.array([[10,5,1]])
    data = np.concatenate((X,Y),axis=0)
    labels = mclust.kMeans(metric="euclidean", heuristic=2, prefix=None).fit(data).labels_
    for i in range(100):
        assert labels[i] == labels[0]
        assert labels[i+100] == labels[100]
