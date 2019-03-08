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
