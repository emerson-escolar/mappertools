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


def compute_flareness(G, member, weight=(lambda v,u,e: 1), query_data='unique_members',verbose=0):
    G_member_nodes, core, shell = compute_member_core_shell(G, member, query_data)

    if len(G_member_nodes) == 0:
        print("member {} not found. Ignoring".format(member))
        return
    
    # assert(G_member_nodes == set(core).union(shell))
    if verbose > 0:
        print("G_member_nodes: ", G_member_nodes)
        print("core: ", core)
        print("shell: ", shell)

    G_member = G.subgraph(G_member_nodes)
    distances = {}
    if len(shell) > 0:
        distances = nx.multi_source_dijkstra_path_length(G_member, shell,weight=weight)

    if verbose > 0:
        print(distances)

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


def compute_all_flareness(G, members, weight=(lambda v,u,e: 1), query_data='unique_members'):
    not_found = []
    not_flare_nor_island = []
    flareness = {}
    for member in members:
        k = compute_flareness(G, member, weight, query_data)
        if k is None:
            not_found.append(member)
        elif len(k) == 0:
            not_flare_nor_island.append(member)
        else:
            flareness[member] = k

    return (flareness, not_flare_nor_island, not_found)


def has_island(k):
    return np.any(np.array(k) == np.inf)

def has_flare(k):
    return np.any(np.array(k) != np.inf)

def is_pure_island(k):
    return len(k) > 0 and np.alltrue(np.array(k) == np.inf)


def flareness_report(G, members, weight=(lambda v,u,e: 1), query_data='unique_members'):
    flareness, not_flare_nor_island, not_found = compute_all_flareness(G,members,weight,query_data)

    pure_island = []
    both = []
    flare = []
    for firm, k in flareness.items():
        if is_pure_island(k):
            pure_island.append(firm)
        elif has_flare(k):
            if has_island(k):
                both.append(firm)
            else:
                flare.append(firm)

    flare = sorted(flare, key=lambda x:max(flareness[x]))

    print("***PURE ISLAND***")
    for x in pure_island:
        print(x)
    print()

    print("***FLARE AND ISLAND***")
    for x in both:
        print(x, flareness[x])
    print()

    print("***FLARE ONLY***")
    for x in flare:
        print(x, flareness[x])
    print()

    print("***NOT FLARE NOR ISLAND***")
    for x in not_flare_nor_island:
        print(x)
    print()


    print("***NOT FOUND***")
    for x in not_found:
        print(x)
