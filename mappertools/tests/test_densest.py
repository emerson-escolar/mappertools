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
    G.add_edge(1,2, weight=3)
    return G

@pytest.fixture
def fragmented():
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4,5])
    return G

def test_small_graph(small_nxgraph):
    assert ds.check_edge_weights(small_nxgraph) == True

def test_no_weight(small_nxgraph):
    small_nxgraph.add_edge(4,5)
    assert ds.check_edge_weights(small_nxgraph) == False

def test_incomparable_weight(small_nxgraph):
    small_nxgraph.add_edge(4,5, weight="foobar")
    assert ds.check_edge_weights(small_nxgraph) == False


def test_negative_weight(small_nxgraph):
    small_nxgraph.add_edge(4,5, weight=-1)
    assert ds.check_edge_weights(small_nxgraph) == False

def test_small_densest(small_nxgraph):
    density, S = ds.graph_peeling(small_nxgraph)
    assert density == 1.5

def test_fragmented(fragmented):
    fragmented.add_edge(-1,0, weight=3)
    density, S = ds.graph_peeling(fragmented)
    assert density == 1.5
