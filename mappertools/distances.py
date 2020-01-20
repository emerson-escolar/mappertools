import numpy as np
import sklearn.preprocessing


def bloom_mahalanobis_closeness(X):
    """
    Compute the Mahalanobis normed technology closeness measure
    as defined in Bloom, Schankerman, Van Reenen (2013)

    This is a similarity measure, not distance!
    High values imply closer observations.

    Parameters
    ----------
    X : array [n_samples, n_features]
        data as a feature array.

    Returns
    -------
    closeness : array [n_samples, n_samples]
                Mahalanobis normed technology closeness
    """

    Xprime = np.matmul(sklearn.preprocessing.normalize(X, axis=1),
                       sklearn.preprocessing.normalize(X.T, axis=1))
    closeness = np.matmul(Xprime, Xprime.T)

    return closeness


def normalize_diagonal_to_one(A):
    """
    Parameters
    ----------
    A : array [n, n]
        square matrix to normalize, with
        all diagonal entries positive

    Returns
    -------
    normed_A : array [n, n]
        normalized square matrix
    """

    L = np.diag(np.sqrt(1./np.diag(A)))
    normed_A = L @ A @ L
    np.fill_diagonal(normed_A, 1)

    return normed_A


def flipped_bloom_mahalanobis_distance(X):
    """
    Compute a distance based on
    Mahalanobis normed technology closeness measure
    as defined in Bloom, Schankerman, Van Reenen (2013)

    Parameters
    ----------
    X : array [n_samples, n_features]
        data as a feature array.

    Returns
    -------
    dissimilarity : array [n_samples, n_samples]
        distance matrix
    """

    closeness = bloom_mahalanobis_closeness(X)
    dissimilarity = np.max(closeness) - closeness
    np.fill_diagonal(dissimilarity, 0)
    assert np.allclose(dissimilarity, dissimilarity.T)

    return (dissimilarity + dissimilarity.T)/2



def normalized_bloom_mahalanobis_distance(X):
    """
    Compute a distance based on
    Mahalanobis normed technology closeness measure
    as defined in Bloom, Schankerman, Van Reenen (2013)

    Parameters
    ----------
    X : array [n_samples, n_features]
        data as a feature array.

    Returns
    -------
    dissimilarity : array [n_samples, n_samples]
        distance matrix
    """

    closeness = bloom_mahalanobis_closeness(X)
    dissimilarity = 1 - normalize_diagonal_to_one(closeness)
    assert np.allclose(dissimilarity, dissimilarity.T)

    return (dissimilarity + dissimilarity.T)/2
