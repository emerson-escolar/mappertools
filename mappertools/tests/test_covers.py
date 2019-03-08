import pytest
import mappertools.covers as cv
import numpy as np

def test_uniform_no_overlap():
    lb, ub = cv.uniform_cover_fences(0,10,2,0)
    assert np.all(lb == np.array([0,5]))
    assert np.all(ub == np.array([5,10]))


def test_uniform_overlapped():
    lb, ub = cv.uniform_cover_fences(0,10,2,0.75)
    assert np.all(lb == np.array([0,2]))
    assert np.all(ub == np.array([8,10]))

def test_uniform_lengths_overlaps():
    p = 0.42
    n = 13
    lb, ub = cv.uniform_cover_fences(0,100, n, p)
    assert len(ub) == n == len(lb)

    lengths = ub - lb
    assert np.allclose(lengths, lengths[0])

    # check that actual overlap is correct
    for i in range(len(lb)-1):
        assert np.isclose( (ub[i] - lb[i+1]), p * (ub[i]-lb[i]))
        assert np.isclose( (ub[i] - lb[i+1]), p * (ub[i+1]-lb[i+1]))
