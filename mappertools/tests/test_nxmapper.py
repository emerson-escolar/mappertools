import pytest

import mappertools.text_dump as td
import networkx as nx
import kmapper as km
import sklearn.cluster



@pytest.fixture
def small_nxgraph():
    G = nx.Graph()
    G.add_node(1,membership=[0,1,3])
    G.add_node(2,membership=[1,2,3])
    G.add_edge(1,2,membership=[1,3])

    extra_data = {'colors' : ['red', 'green', 'blue', 'black'],
                  'day' : {0:10,1:20,2:30,3:40}}
    transforms = {'colors': (lambda x : "".join(x)), 'day' : min}

    return G, extra_data, transforms


def test_nxmapper_node_data(small_nxgraph):
    G,extra_data,_ = small_nxgraph
    td.nxmapper_append_node_member_data(G, extra_data)

    assert (G.nodes[1]['colors'] == ['red','green','black'])
    assert (G.nodes[2]['colors'] == ['green','blue','black'])

    assert (G.nodes[1]['day'] == [10,20,40])
    assert (G.nodes[2]['day'] == [20,30,40])


def test_nxmapper_node_data_transformed(small_nxgraph):
    G,extra_data,transforms = small_nxgraph
    td.nxmapper_append_node_member_data(G, extra_data, transforms)

    assert (G.nodes[1]['colors'] == 'redgreenblack')
    assert (G.nodes[2]['colors'] == 'greenblueblack')

    assert (G.nodes[1]['day'] == 10)
    assert (G.nodes[2]['day'] == 20)


def test_nxmapper_edge_data(small_nxgraph):
    G,extra_data,_ = small_nxgraph
    td.nxmapper_append_edge_member_data(G, extra_data)

    assert (G.edges[(1,2)]['colors'] == ['green','black'])
    assert (G.edges[(1,2)]['day'] == [20,40])


def test_nxmapper_edge_data_transformed(small_nxgraph):
    G,extra_data,transforms = small_nxgraph
    td.nxmapper_append_edge_member_data(G, extra_data, transforms)

    assert (G.edges[(1,2)]['colors'] == 'greenblack')
    assert (G.edges[(1,2)]['day'] == 20)




def test_kmapper_sample():
    data = km.np.array([[0],[1],[2]])
    lens = data

    graph = km.KeplerMapper().map(data, data, clusterer=sklearn.cluster.DBSCAN(eps=1, min_samples=0),
                                  cover=km.Cover(n_cubes=2, perc_overlap=1))
    nxgraph = td.kmapper_to_nxmapper(graph)
    assert len(nxgraph.edges) == 1
    assert len(nxgraph.nodes) == 2

    for _,_,data in nxgraph.edges.data():
        assert 'membership' in data

    for _,data in nxgraph.nodes.data():
        assert 'membership' in data
