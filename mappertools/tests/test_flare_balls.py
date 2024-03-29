import pytest
import mappertools.features.core as fc
import mappertools.features.flare_balls as fb
import networkx as nx

import numpy as np
import math



def test_core_shell_path_graph():
    G = nx.path_graph(5)
    H = {0,1,2}

    core,shell = fc.compute_core_shell(G,H)

    assert set(core) == {0,1}
    assert set(shell) == {2}


def test_core_shell_pure_island():
    G = nx.path_graph(5)

    core,shell = fc.compute_core_shell(G,G)

    assert set(core) == set(G.nodes)
    assert len(shell) == 0



def test_flare_signature():
    test_data = (([1,1,2],1,2),
                 ([np.inf],3,np.inf),
                 ([math.inf],3,math.inf),
                 ([3,4,np.inf],2,4),
                 ([],0,0))

    for k,k_type,k_index in test_data:
        comp_type, comp_index = fb.flare_type_index(k)
        assert k_type == comp_type
        assert k_index == comp_index


def test_flareness():
    G = nx.path_graph(6)
    for node in G.nodes:
        G.nodes[node]['unique_members'] = ['foo']
    G.nodes[0]['unique_members'] = ['bar']

    # vertices are as follows
    # core: 5, 4, 3, 2
    # shell: 1
    # outside: 0

    k, components = fb.compute_flareness(G, 'foo')
    assert len(k) == 1

    # length 4 path to exit core, 5 to 1
    assert k[0] == 4

def test_island():
    G = nx.path_graph(6)
    for node in G.nodes:
        G.nodes[node]['unique_members'] = ['foo']

    k, components = fb.compute_flareness(G, 'foo')
    assert len(k) == 1
    assert k[0] == np.inf


def test_flare_and_island():
    G = nx.disjoint_union(nx.path_graph(6), nx.path_graph(4))
    for node in G.nodes:
        G.nodes[node]['unique_members'] = ['foo']
    G.nodes[0]['unique_members'] = ['bar']

    k, components = fb.compute_flareness(G, 'foo')
    assert len(k) == 2

    assert set(k) == {4, np.inf}
