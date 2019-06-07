import networkx as nx
import numpy as np

def get_nodes_containing_member(G, member, query_data='unique_members'):
    return (node for node in G if member in G.nodes[node][query_data])


def compute_member_core_shell(G, member, query_data='unique_members'):
    member_core = []
    member_shell = []

    G_member = set(get_nodes_containing_member(G, member, query_data))
    for x in G_member:
        for y in G[x]:
            if y not in G_member:
                member_shell.append(x)
                break
        else:
            # yes this is not a bug. 'else' is aligned with 'for'
            member_core.append(x)

    return G_member, member_core, member_shell


def compute_flareness(G, member, weight=(lambda v,u,e: 1), query_data='unique_members'):
    G_member_nodes, core, shell = compute_member_core_shell(G, member, query_data)

    if len(G_member_nodes) == 0:
        print("member {} not found. Ignoring".format(member))
        return
    
    # assert(G_member_nodes == set(core).union(shell))
    # print("G_member_nodes: ", G_member_nodes)
    # print("core: ", core)
    # print("shell: ", shell)

    G_member = G.subgraph(G_member_nodes)
    distances = {}
    if len(shell) > 1:
        distances = nx.multi_source_dijkstra_path_length(G_member, shell,weight=weight)

    k = []
    for L in nx.connected_components(G.subgraph(core)):
        k_L = -np.inf
        for x in L:
            if x in distances and distances[x] > k_L:
                k_L = distances[x]
            elif x not in distances:
                k_L = np.inf
                break
        k.append(k_L)
    return k
