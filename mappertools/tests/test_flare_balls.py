import pytest
import mappertools.features.flare_balls as fb
import networkx as nx

import numpy as np
import math



def test_core_shell_path_graph():
    G = nx.path_graph(5)
    H = {0,1,2}

    core,shell = fb.compute_core_shell(G,H)

    assert set(core) == {0,1}
    assert set(shell) == {2}


def test_core_shell_pure_island():
    G = nx.path_graph(5)

    core,shell = fb.compute_core_shell(G,G)

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
