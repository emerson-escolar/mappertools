import pytest
import mappertools.features.flares as flr

def test_flare():
    x = flr.Flare(0,10)

    x.terminate(1,20)
    assert x.finished == True
    assert 0 in x.nodes
    assert 1 not in x.nodes
