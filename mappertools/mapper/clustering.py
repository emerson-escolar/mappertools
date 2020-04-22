import math
import numpy as np
import pandas

import sklearn.base
import scipy.spatial.distance as ssd

import pyclustering.cluster.kmedoids as kmedoids
import pyclustering.cluster.kmeans as kmeans

import pyclustering.cluster.center_initializer as pci
import pyclustering.utils.metric as pcm

pyclustering_metrics = {"euclidean": pcm.type_metric.EUCLIDEAN,
                        "euclidean_square": pcm.type_metric.EUCLIDEAN_SQUARE,
                        "manhattan": pcm.type_metric.MANHATTAN,
                        "chebyshev": pcm.type_metric.CHEBYSHEV,
                        "minkowski": pcm.type_metric.MINKOWSKI,
                        "canberra": pcm.type_metric.CANBERRA,
                        "chi_square": pcm.type_metric.CHI_SQUARE,
                        "gower": pcm.type_metric.GOWER}
scipy_metrics = {"braycurtis": ssd.braycurtis,
                 "correlation": ssd.correlation,
                 "cosine": ssd.cosine,
                 "jensenshannon": ssd.jensenshannon}

def _process_metric(metric):
    if metric == "precomputed":
        return None

    if metric in pyclustering_metrics:
        pcc_type_metric = pyclustering_metrics[metric]
        return pcm.distance_metric(pcc_type_metric)
    if callable(metric):
        pcc_type_metric = pcm.type_metric.USER_DEFINED
        metric_func = metric
    elif metric in scipy_metrics:
        pcc_type_metric = pcm.type_metric.USER_DEFINED
        metric_func = scipy_metrics[metric]
    else:
        raise RuntimeError("Metric {} not recognized".format(str(metric)))

    pcc_metric = pcm.distance_metric(pcc_type_metric, func = metric_func)
    return pcc_metric

def _clusters_to_labels(clusters, prefix=None):
    """
    Convert clusters to labels

    Parameters
    ----------
    clusters : list of lists
        A list of clusters, each cluster expressed as a list of member indices
        No checking is done for shared members.

    prefix : str
        Optional string to prefix labels with.

    Returns
    -------
    labels : list of str
        List of labels, according to member index.
        Label is given by 'prefix_clusternumber', where clusternumber
        is the last cluster in clusters which contains index.
    """

    n_clusters = len(clusters)
    if n_clusters == 0: return []

    fstr = prefix + "_" if prefix else ""
    fstr += ("{:0" + str(math.ceil(math.log(n_clusters,10))) + "d}")

    n_points = sum([len(cluster) for cluster in clusters])
    labels = ["none"] * n_points
    for i, cluster in enumerate(clusters):
        cluster_str = fstr.format(i)
        for point in cluster:
            labels[point] = cluster_str

    return labels


class _kType(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    """
    Base class for wrappers around pyclustering.cluster

    Emulates classes in sklearn.cluster, for compatibility with kmapper.
    """
    def __init__(self, metric, heuristic, k_max=None, prefix="", verbose=1):
        self.metric = metric
        self.pcc_metric = _process_metric(metric)

        self.heuristic = heuristic
        self.verbose = verbose

        self.prefix = prefix

        if k_max == None:
            k_max = np.inf
        self.k_max = k_max

    @staticmethod
    def _validate_data(X):
        if hasattr(X, "to_numpy") and callable(X.to_numpy):
            X = X.to_numpy()
        return X

    def _fit_k(self, X, k):
        raise NotImplementedError

    def fit(self, X, y=None):
        if isinstance(self.heuristic, int):
            return self._fit_k(X, self.heuristic)


class kMedoids(_kType):
    """
    kMedoids clustering

    Just a wrapper around pyclustering.cluster.kmedoids
    emulating sklearn.cluster classes, for compatibility with kmapper.
    """
    def __init__(self, metric, heuristic, k_max=None, prefix="kMedoids", verbose=1):
        super().__init__(metric, heuristic, k_max, prefix, verbose)

    def _fit_k(self, X, k):
        X = self._validate_data(X)
        initial_medoids = pci.random_center_initializer(X, k).initialize(return_index=True)

        if self.metric == "precomputed":
            ans = kmedoids.kmedoids(X, initial_medoids, data_type = "distance_matrix")
        else:
            ans = kmedoids.kmedoids(X, initial_medoids, metric=self.pcc_metric)
        ans.process()
        self.labels_ = _clusters_to_labels(ans.get_clusters(), prefix=self.prefix)

        return self


# buggy:
# Problem: pyclustering kMeans does not support properly custom distances:
# Both precomputed matrix and pcm.type_metric.USER_DEFINED function
class kMeans(_kType):
    def __init__(self, metric, heuristic, k_max=None, prefix="kMeans", verbose=1):
        if metric != "euclidean":
            raise RuntimeError("kMeans only allowed for Euclidean metric")
        
        super().__init__(metric, heuristic, k_max, prefix, verbose)

    def _fit_k(self, X, k):
        X = self._validate_data(X)
        initial_centers = pci.kmeans_plusplus_initializer(X, k).initialize()

        if self.metric == "precomputed":
            ans = kmeans.kmeans(X, initial_centers, data_type = "distance_matrix")
        else:
            ans = kmeans.kmeans(X, initial_centers, metric=self.pcc_metric)
        ans.process()
        self.labels_ = _clusters_to_labels(ans.get_clusters(), prefix=self.prefix)

        return self


def unique_entity_counts_by_cluster(labels, unique_names=None):
    """
    Convert labels to boolean matrix then optionally aggregate by unique names
    """
    ans = pandas.get_dummies(labels)
    if unique_names is not None:
        ans = ans.groupby(unique_names).sum()
    return ans
