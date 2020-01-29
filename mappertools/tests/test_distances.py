import pytest
import mappertools.distances as dst
import numpy as np
import scipy.spatial.distance
import sklearn.preprocessing

def test_no_overlap():
    """
    Input X is firm x patentclass matrix.

    In this example, each firm patents in exactly one patent class,
    so, no patent class overlaps with any other patent class in the same firm.

    In this case, it is known that Bloom et al's closeness is the same as
    Jaffe's measure, which is 1 - cosine distance.
    """
    X = np.array([[1,0,0,0,0,0],
                  [0,3,0,0,0,0],
                  [0,2,0,0,0,0],
                  [0,0,0,0,0,7]])

    bloom = dst.bloom_mahalanobis_closeness(X)
    cos = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric='cosine'))

    assert np.allclose(bloom, 1-cos)
