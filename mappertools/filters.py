import numpy
from scipy.spatial import distance


def laplacian_gauss_kernel(X, epsilon, metric='euclidean'):
    dists = distance.pdist(X, metric)
    weights = numpy.exp(numpy.square(dists) * (-1/epsilon))

    return laplacian_from_weights(distance.squareform(weights))

def laplacian_from_weights(weights):
    k0 = weights.sum(axis=0, keepdims=True)
    k1 = weights.sum(axis=1, keepdims=True)

    return (weights/numpy.sqrt(k0))/numpy.sqrt(k1)





def gauss_kernel_density(X, epsilon, metric='euclidean'):
    """
    Gauss kernel density estimation from observations in n-dimensional space

    Parameters
    ----------
    X : ndarray
        An m by n array of m original observations in an n-dimensional space.
    epsilon: float
        Parameter for Gauss kernel. Equal to twice sigma squared.
    metric: str or function, optional
        The distance metric to use. The distance function can
        be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'kulsinski', 'mahalanobis', 'matching',
        'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
        (See scipy.spatial.distance.pdist)
    """

    return gauss_kernel_density_from_dist(distance.pdist(X, metric), epsilon)


def gauss_kernel_density_from_dist(dists, epsilon):
    """
    Gauss kernel density estimation, given pairwise distances.

    Parameters
    ----------
    dists : ndarray
        Either a condensed or redundant distance matrix.
        (see scipy.spatial.distance)
    epsilon: float
        epsilon: float
        Parameter for Gauss kernel. Equal to twice sigma squared.
    """

    d = numpy.exp(numpy.square(dists) * (-1/epsilon))
    s = d.shape

    if len(s) == 1:
        ans = distance.squareform(d)
        numpy.fill_diagonal(ans,1)
    elif len(s) == 2:
        ans = d
    else:
        raise ValueError('The first argument must be one or two dimensional array. A %d-dimensional array is not permitted' % len(s))

    return numpy.sum(ans, axis=1,keepdims=True)/numpy.sum(ans)



def eccentricity_from_dist(dists, p=2):
    """
    Compute eccentricity

    Parameters
    ----------
    dists : ndarray
        Either a condensed or redundant distance matrix.
        (see scipy.spatial.distance)
    p: positive integer, or np.inf
        order of exponent
    """

    s = dists.shape
    if len(s) == 1:
        ans = distance.squareform(dists)
    elif len(s) == 2:
        ans = dists
    else:
        raise ValueError('The first argument must be one or two dimensional array. A %d-dimensional array is not permitted' % len(s))

    return numpy.linalg.norm(ans, ord=p, axis=1,keepdims=True) / (ans.shape[0] ** (1./p))

def eccentricity(X, p=2, metric='euclidean'):
    return eccentricity_from_dist(distance.pdist(X,metric), p)
