







def get_nodes_containing_entity(G, entity, query_data='unique_members'):
    """
    Given a Mapper graph, where each node is a set of observations of
    different entities, we find the ndoes containing a given entity.

    That is, each entity may appear more than once as different observations in different nodes.


    Parameters
    ----------
    G : networkx graph
        The Mapper graph to query.

    entity :
        The particular entity we want to find in G.

    query_data : str
        node attribute key containing 'unique members' (names of entities) of each node

    Returns
    -------
    nodes : list
        list of nodes containing at least one observation of entity
    """

    nodes = (node for node in G if entity in G.nodes[node][query_data])
    return nodes


def compute_core_shell(G, H):
    """
    Compute core-shell decomposition of a subset of nodes H with respect to graph G.

    A vertex x in H is said to be core if all neighbors of x are in H.
    Otherwise, it is said to be shell.

    Parameters
    ----------
    G : networkx graph

    H : iterable of nodes in G

    Returns
    -------
    (core, shell) : tuple of lists
        list of vertices in the core and shell, respectively
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
