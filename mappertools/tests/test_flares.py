import pytest
import mappertools.features.flares as flr
import networkx as nx

def test_flare():
    x = flr.Flare(0,10)

    x.terminate(1,20)
    assert x.finished == True
    assert 0 in x.nodes
    assert 1 not in x.nodes


def test_detection():
    G = nx.generators.classic.path_graph(10)

    # trivial 'centrality'
    cen = {n:1 for n in G.nodes}
    flares = flr.flare_detect(G,cen)
    assert len(flares) == 1
    assert len(flares[0].nodes) == 10

    # harmonic centrality
    cen = nx.centrality.harmonic_centrality(G)
    flares = flr.flare_detect(G,cen)
    assert len(flares) == 2
