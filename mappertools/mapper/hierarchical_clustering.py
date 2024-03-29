import warnings
import numpy as np

import sklearn.base
import sklearn.decomposition
# import sklearn.cluster

import scipy.cluster.hierarchy
import scipy.spatial.distance

from mappertools.mapper.clustering_scores import negative_silhouette

def cluster_number_to_threshold(k, merge_distances):
    # check merge distances is non decreasing:
    assert np.all(np.diff(merge_distances) >= 0)

    # threshold is kth entry counting from the last.
    return (merge_distances[-k] if k <= len(merge_distances) else -np.inf)


def find_histogram_gap(merge_distances, percentile, bins='doane'):
    hist, bin_edges = np.histogram(merge_distances, bins=bins)

    if np.all(hist != 0):
        return None

    gaps = np.argwhere(hist == 0).flatten()
    idx = np.percentile(gaps, percentile, method='nearest')
    threshold = bin_edges[idx]
    return threshold


def mapper_gap_heuristic(Z, percentile, k_max=None, bins="doane"):
    """
    Parameters
    ----------
    Z : array
        hierarchical clustering encoded as a linkage matrix.
        Output of scipy.cluster.hierarchy.linkage applied to X with given metric.
        It is user responsibility to assure that this is indeed the case.

    percentile : int in [0,100]
        Percentile gap to use. 0 corresponds to first gap

    k_max : int, optional
        Maximum number of clusters.

    bins : int or string, optional
        Number of bins to use for creating the histogram of merge-distances.
        Either a positive integer, or choose from:
        ‘auto’, ‘fd’, ‘doane’, ‘scott’, ‘stone’, ‘rice’, ‘sturges’, ‘sqrt’.

        Internally uses numpy.histogram.
    """
    merge_distances = Z[:,2]

    if k_max != None and k_max != np.inf:
        merge_distances = merge_distances[-k_max:]

    threshold = find_histogram_gap(merge_distances,percentile, bins)
    if threshold == None:
        labels = np.ones(Z.shape[0]+1)
        k = 1
    else:
        labels = scipy.cluster.hierarchy.fcluster(Z, t=threshold, criterion='distance')
        k = len(set(labels))

    return labels, k



def statistic_heuristic_hierarchical(X, metric, Z,
                                     k_max, statistic=negative_silhouette):
    """
    Hierarchical clustering thresholding by statistic

    Determine 'best' clustering by minimizing value of statistic function
    over different thresholding of the hierarchical clustering Z.

    Brute force search.

    Parameters
    ----------
    X : array [n_samples, n_samples] if metric == "precomputed", or, \
             [n_samples, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    metric : str or function, optional
        See the ``scipy.spatial.distance.pdist`` function for a list of valid distance metrics.
        A custom distance function can also be used. If X is the distance
        array itself, use ``metric="precomputed"``.

    Z : array
        hierarchical clustering encoded as a linkage matrix.
        Output of scipy.cluster.hierarchy.linkage applied to X with given metric.
        It is user responsibility to assure that this is indeed the case.

    k_max : int, optional
        Maximum number of clusters.

    statistic : function with signature (X,labels,metric -> statistic_value)
        Statistic function that evaluates the 'goodness' of clustering given by labels.
        Smaller values are interpreted as better.
    """
    # N data points imply length N-1 merge_distances
    # statistic-based heuristic searches over 2 <= k <= N-1
    # to avoid trivial clustering.
    merge_distances = Z[:,2]
    N = len(merge_distances) + 1

    optimal_stat = np.inf
    optimal_labels, optimal_k =  np.array([1]*N), 1

    for threshold in reversed(np.sort(np.unique(merge_distances))):
        labels = scipy.cluster.hierarchy.fcluster(Z, t=threshold, criterion='distance')
        cur_k = len(np.unique(labels))
        if cur_k < 2:
            continue
        if cur_k > k_max or cur_k > N-1:
            break
        cur_stat = statistic(X, labels, metric)
        if cur_stat <  optimal_stat:
            optimal_stat = cur_stat
            optimal_labels, optimal_k = labels, cur_k

    return optimal_labels, optimal_k


class PreTransformPCA(object):
    """
    Projection onto chosen PCA axes.

    For use as pre_transform in HeuristicHierarchical.
    """
    def __init__(self, pc_axes, precomputed=None):
        self.pc_axes = pc_axes
        self.precomputed = precomputed

    def transform(self, X):
        if X.shape[0]==1:
            return X

        if self.precomputed != None:
            pca = self.precomputed
        else:
            pca = sklearn.decomposition.PCA(n_components=np.max(self.pc_axes)+1)
            pca.fit(X)
        return pca.transform(X)[:,self.pc_axes]


class HeuristicHierarchical(sklearn.base.BaseEstimator,
                            sklearn.base.ClusterMixin):
    """
    Hierarchical Clustering, using heuristic to determine clusters

    Uses scipy.cluster.hierarchy algorithms.
    Class is designed to emulate sklearn.cluster format, for compatibility with kmapper.

    Instead of having to specify n_clusters, uses some heuristic to determine number of clusters.

    Parameters
    ----------
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}
        Which linkage method to use, as available in scipy.cluster.hierarchy.linkage.

    metric : str or function, optional
        See the ``scipy.spatial.distance.pdist`` function for a list of valid distance metrics.
        A custom distance function can also be used.

    heuristic : {"firstgap", "midgap", "lastgap", "silhouette"}
        Which heuristic to use to determine number of clusters.
        first/mid/last gap is based on the original Mapper paper.
        silhouette uses the silhouette score.

    bins :
        Which heuristic to use for the histogram binning in the
        "gap"-based heuristics (firstgap, midgap, lastgap).
        Refer to numpy.histogram_bin_edges for choices.

    pre_transform :
        A class instance with a transform method.
        This is applied to the input when HeuristicHierarchical.fit
        is called. Not compatible with a precomputed metric!

    k_max : int, optional
        Maximum number of clusters.

    Attributes
    ----------
    labels_ : array [n_samples]
        cluster labels for each point

    Notes
    -----
    This is essentially a wrapper around scipy.cluster.hierarchy.linkage.
    In particular, warnings about that algorithm apply here too.

    Quoted:
    "Methods 'centroid', 'median' and 'ward' are correctly defined only if
    Euclidean pairwise metric is used. If `X` is passed as precomputed
    pairwise distances, then it is a user responsibility to assure that
    these distances are in fact Euclidean, otherwise the produced result
    will be incorrect."
    """

    def __init__(self, method='single', metric='euclidean', heuristic='firstgap',
                 bins='doane', pre_transform = None, k_max=None, verbose=1):
        self.method = method
        self.metric = metric
        self.heuristic = heuristic
        self.bins = bins
        self.verbose = verbose
        self.pre_transform = pre_transform

        print("Clustering using: Hierarchical clustering with " + method + " linkage and " + heuristic + " heuristic.")

        if k_max == None:
            k_max = np.inf
        self.k_max = k_max

        if self.metric == 'precomputed' and self.pre_transform != None:
            raise RuntimeError("Using pre_transform not valid with precomputed metric!")


    def fit(self, X, y=None):
        """Fit the HeuristicHierarchical clustering on data

        Parameters
        ----------
        X : array [n_samples, n_samples] if metric == "precomputed", or, \
             [n_samples, n_features] otherwise
            Array of pairwise distances between samples, or a feature array.

        y : ignored

        Returns
        -------
        self
        """

        if self.pre_transform != None:
            X = self.pre_transform.transform(X)

        if len(X.shape) == 2 and X.shape[0] == 1:
            if self.verbose > 0:
                print("1 clusters detected in {} points".format(X.shape[0]))
            self.labels_ = np.array([1])
            return

        if self.metric != 'precomputed':
            Z = scipy.cluster.hierarchy.linkage(X, method=self.method, metric=self.metric)
        else:
            #flatten
            compdists = scipy.spatial.distance.squareform(X, force='tovector')
            Z = scipy.cluster.hierarchy.linkage(compdists, method=self.method, metric=self.metric)

        if self.verbose >= 2:
            print("*** Heuristic Hierarchical Clustering Report ***")
            if X.shape[0] > 2:
                if self.metric != 'precomputed':
                    dists = scipy.spatial.distance.pdist(X, metric=self.metric)
                else:
                    dists = compdists
                c, _ = scipy.cluster.hierarchy.cophenet(Z, dists)
                print("cophentic correlation distance: {}".format(c))
            else:
                print("cophentic correlation distance: invalid, too few data points")

        # MAPPER PAPER GAP HEURISTIC
        gap_heuristic_percentiles = {'firstgap': 0, 'midgap': 50, 'lastgap':100}
        if self.heuristic in gap_heuristic_percentiles:
            percentile = gap_heuristic_percentiles[self.heuristic]
            self.labels_, k = mapper_gap_heuristic(Z, percentile, self.k_max, self.bins)

        elif self.heuristic == 'sil' or self.heuristic == 'silhouette':
            self.labels_, k = statistic_heuristic_hierarchical(X, self.metric, Z, self.k_max, statistic=negative_silhouette)
        else:
            pass

        # FINAL REPORTING
        if self.verbose > 0:
            print("{} clusters detected in {} points".format(k,X.shape[0]))

        if self.verbose >= 2:
            if k <= 1:
                print("silhouette score: invalid, too few final clusters")
            else:
                print("silhouette score: {}".format(-negative_silhouette(X, self.labels_, metric=self.metric)))
        return self


class LinkageMapper(HeuristicHierarchical):
    def __init__(self, method='single', metric='euclidean', heuristic='firstgap',
                 bins='doane', pre_transform = None, k_max=None, verbose=1):
        warnings.warn("LinkageMapper is planned to be deprecated. Use HeuristicHierarchical instead.", PendingDeprecationWarning)
        super().__init__(method=method, metric=metric, heuristic=heuristic,
                         bins=bins, pre_transform=pre_transform, k_max=k_max, verbose=verbose)
