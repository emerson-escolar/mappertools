import pytest
import mappertools.features.flare_balls as fb
import networkx as nx




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
