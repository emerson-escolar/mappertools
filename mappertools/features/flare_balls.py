import networkx as nx
import numpy as np

import pandas

def get_nodes_containing_member(G, member, query_data='unique_members'):
    return (node for node in G if member in G.nodes[node][query_data])


def compute_core_shell(G, H):
    """
    Compute core-shell decomposition of a subset of nodes H with respect to graph G.

    A vertex x in H is said to be core if all neighbors of x are in H.
    Otherwise, it is said to be shell
    """

    core = []
    shell = []
    for x in H:
        for y in G[x]:
            if y not in H:
                shell.append(x)
                break
        else:
            # yes this is not a bug. 'else' is aligned with 'for'
            core.append(x)

    return core, shell


def compute_flareness(G, member,
                      weight=(lambda v,u,e: 1), query_data='unique_members',
                      verbose=0):
    G_member = set(get_nodes_containing_member(G, member, query_data))
    core, shell = compute_core_shell(G, G_member)

    if len(G_member) == 0:
        if verbose > 0: print("Member {} not found. Ignoring".format(member))
        return None, None

    if verbose > 0:
        print("G_member: ", G_member)
        print("core: ", core)
        print("shell: ", shell)

    G_member_subgraph = G.subgraph(G_member)
    distances = {}
    if len(shell) > 0:
        distances = nx.multi_source_dijkstra_path_length(G_member_subgraph,shell,
                                                         weight=weight)
    k = []
    components = list(nx.connected_components(G.subgraph(core)))
    for L in components:
        k_L = max(distances.get(x,np.inf) for x in L)
        k.append(k_L)

    return (k, components)




def compute_all_summary(G, members, weight=(lambda v,u,e: 1),
                        query_data='unique_members', verbose=0,
                        keep_missing=False):
    ans = pandas.DataFrame(columns=['type','k_C','k_vec'],index=members)
    for member in members:
        k, _ = compute_flareness(G, member, weight, query_data, verbose)
        if k is None and keep_missing:
            ans.loc[member] = pandas.Series({'type':-1, 'k_C':None, 'k_vec':None})
        else:
            k_type, k_index = flare_type_index(k)
            ans.loc[member] = pandas.Series({'type':k_type, 'k_C':k_index, 'k_vec':k})

    return ans


def has_island(k):
    return np.any(np.array(k) == np.inf)

def has_flare(k):
    return np.any(np.array(k) != np.inf)

def is_pure_island(k):
    return len(k) > 0 and np.alltrue(np.array(k) == np.inf)

def finmax(k):
    return max(np.array(k)[np.array(k) != np.inf])

def flare_type_index(flare_signature):
    if len(flare_signature) == 0:
        k_type = 0
        k_index = 0
    elif is_pure_island(flare_signature):
        k_type = 3
        k_index = np.inf
    else:
        # not pure island and nonempty implies has flare
        assert has_flare(flare_signature)
        k_index = finmax(flare_signature)
        if has_island(flare_signature):
            k_type = 2
        else:
            k_type = 1
    return (k_type, k_index)
