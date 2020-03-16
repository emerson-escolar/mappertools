import pytest
import scipy.cluster.hierarchy
import scipy.spatial.distance as spd
import numpy as np

import mappertools.mapper.hierarchical_clustering as hc

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


def test_num_clusters():
    X = np.random.normal(size=(400,2))

    Z = scipy.cluster.hierarchy.linkage(X, method='single', metric='euclidean')
    merge_distances = Z[:,2]

    for i,t in enumerate(reversed(merge_distances)):
        labels = scipy.cluster.hierarchy.fcluster(Z, t, criterion='distance')

        assert len(set(labels)) == i+1
        assert t == hc.cluster_number_to_threshold(len(set(labels)), merge_distances)


def test_heuristics():
    fg = hc.LinkageMapper(heuristic='firstgap').fit(X)
    assert len(np.unique(fg.labels_)) == 3

    sil = hc.LinkageMapper(heuristic='sil').fit(X)
    assert len(np.unique(sil.labels_)) == 3


def test_heuristics_precomputed():
    dists = spd.squareform(spd.pdist(X))

    fg = hc.LinkageMapper(heuristic='firstgap', metric='precomputed').fit(dists)
    assert len(np.unique(fg.labels_)) == 3

    sil = hc.LinkageMapper(heuristic='sil', metric='precomputed').fit(dists)
    assert len(np.unique(sil.labels_)) == 3



def test_local_PCA():
    # check that we're just doing pca

    X_local = hc.PreTransformPCA(pc_axes=[0,1]).transform(X)

    actual_pca = hc.sklearn.decomposition.PCA(2).fit(X)
    X_pca = actual_pca.transform(X)

    X_precomputed = hc.PreTransformPCA(pc_axes=[0,1], precomputed=actual_pca).transform(X)

    assert np.allclose(X_local, X_pca)
    assert np.allclose(X_pca, X_precomputed)



def test_local_PCA_with_clustering():
    pt = hc.PreTransformPCA(pc_axes=[0,1])

    fg = hc.LinkageMapper(heuristic='firstgap', pre_transform=pt).fit(X)
    assert len(np.unique(fg.labels_)) == 3

    sil = hc.LinkageMapper(heuristic='sil', pre_transform=pt).fit(X)
    assert len(np.unique(sil.labels_)) == 3
