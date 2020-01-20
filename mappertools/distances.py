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

    return normed_A
