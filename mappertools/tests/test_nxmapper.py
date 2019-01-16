import pytest

import mappertools.text_dump as td
import networkx as nx


@pytest.fixture
def small_nxgraph():
    G = nx.Graph()
    G.add_node(1,membership=[0,1,3])
    G.add_node(2,membership=[1,2,3])

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
