import sklearn.cluster
import sklearn.base
import numpy
import scipy.cluster.hierarchy
import scipy.spatial.distance

import matplotlib.pyplot as plt


class LinkageMapper(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    """
    Agglomerative Linkage Clustering

    Instead of specifying n_clusters, uses the "first-gap" heuristic in the original
    Mapper paper to decide.
    Uses scipy.cluster.hierarchy algorithms.
    Class is designed to emulate sklearn.cluster format, for compatibility with kmapper.

    Parameters
    ----------
    method : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}
        Which linkage method to use, as available in scipy.cluster.hierarchy.linkage.

    metric : str or function, optional
        If y is a collection of observation vectors, distance metric to use.
        Ignored otherwise.
        See the ``scipy.spatial.distance.pdist`` function for a list of valid distance metrics.
        A custom distance function can also be used.

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
    Euclidean pairwise metric is used. If `y` is passed as precomputed
    pairwise distances, then it is a user responsibility to assure that
    these distances are in fact Euclidean, otherwise the produced result
    will be incorrect."
    """

    def __init__(self, method='single', metric='euclidean'):
        self.method = method
        self.metric = metric
        pass

    def fit(self, X, y=None):
        """Fit the Linkage clustering on data

        Parameters
        ----------
        X : ndarray
            A condensed distance matrix. A condensed distance matrix
            is a flat array containing the upper triangular of the distance matrix.
            This is the form that ``pdist`` returns. All elements of the condensed
            distance matrix must be finite, i.e. no NaNs or infs.
            Alternatively, a collection of `m` observation vectors in `n` dimensions
            as an `m` by `n` array.

            If self.metric is a distance matrix, overrides behaviour and uses that distance instead!

        y : ignored

        Returns
        -------
        self
        """

        if len(X.shape) == 2 and X.shape[0] == 1:
            self.labels_ = numpy.array([1])
            return


        Z = scipy.cluster.hierarchy.linkage(X, method=self.method, metric=self.metric)


        merge_distances = Z[:,2]
        hist, bin_edges = numpy.histogram(merge_distances, bins='doane')

        # plt.figure()
        # plt.hist(merge_distances, bins=len(hist))
        # plt.show()

        # first gap heuristic, as in Mapper paper
        if numpy.alltrue(hist != 0):
            # print("no gap!")
            self.labels_ = numpy.ones(Z.shape[0]+1)
            return

        idx = (hist == 0).argmax()
        threshold = bin_edges[idx]

        self.labels_ = scipy.cluster.hierarchy.fcluster(Z, t=threshold, criterion='distance')
