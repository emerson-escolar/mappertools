import pytest

import mappertools.densest_subgraph as ds
import networkx as nx
import kmapper as km
import sklearn.cluster


@pytest.fixture
def small_nxgraph():
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1,2, weight=2)
    return G

def test_no_weight(small_nxgraph):
    G = small_nxgraph
    assert ds.check_edge_weights(G) == True
    G.add_edge(4,5)
    assert ds.check_edge_weights(G) == False

def test_incomparable_weight(small_nxgraph):
    G = small_nxgraph
    G.add_edge(4,5, weight="foobar")
    assert ds.check_edge_weights(G) == False


def test_negative_weight(small_nxgraph):
    G = small_nxgraph
    G.add_edge(4,5, weight=-1)
    assert ds.check_edge_weights(G) == False
