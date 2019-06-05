import pytest
import mappertools.features.flare_tree as flr
import networkx as nx


def test_path_flares():
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

def test_star_flares():
    G = nx.generators.classic.star_graph(10)

    # harmonic centrality
    cen = nx.centrality.harmonic_centrality(G)
    flares = flr.flare_detect(G,cen)
    assert len(flares) == 10
