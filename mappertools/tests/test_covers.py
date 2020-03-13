import pytest
import mappertools.mapper.covers as cv
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


def test_epcover_1d():
    N = 100
    data = np.random.normal(size=N)
    # append ids
    ids = np.arange(N)
    data = np.c_[ids, data]

    cov = cv.EPCover(10,0.5)
    cov.fit(data)
    patches = cov.transform(data)
    member_counts = [len(x) for x in patches]

    assert np.all(np.equal(member_counts, member_counts[0]))


def test_epcover_2d():
    N = 1000
    data = np.random.normal(size=(N,2))
    # append ids
    ids = np.arange(N)
    data = np.c_[ids, data]

    cov = cv.EPCover(13,0.42)
    cov.fit(data)
