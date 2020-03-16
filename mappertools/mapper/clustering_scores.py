import sklearn.metrics

# def DaviesBouldinIndex(X, labels, metric):
#     # for lbl in np.unique(labels):
#     #     print(X[labels==lbl])

#     return 0


def negative_silhouette(X, labels, metric):
    """
    Compute the negative of the silhouette score.
    Uses sklearn.metrics.negative_silhouette

    Parameters
    ----------
    X : array [n_samples, n_samples] if metric == "precomputed", or, \
             [n_samples, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    labels : array, shape = [n_samples]
         Predicted labels for each sample.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`metrics.pairwise.pairwise_distances
        <sklearn.metrics.pairwise.pairwise_distances>`. If X is the distance
        array itself, use ``metric="precomputed"``.
    """

    return -sklearn.metrics.silhouette_score(X, labels, metric=metric)
