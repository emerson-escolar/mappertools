import networkx as nx
import numpy as np

import pandas

import mappertools.features.core as mfc

def compute_flareness(G, entity,
                      weight=(lambda v,u,e: 1), query_data='unique_members',
                      verbose=0):
    """
    Compute "flareness" of entity in Mapper graph G using the proposed definition in
    Escolar et al., "Mapping Firms' Locations in Technological Space"


    See mappertools.features.core.get_nodes_containing_entity for a discussion on entities.

    Parameters
    ----------
    G : networkx graph
        Represents a Mapper graph, where each node may be
        a set of observations of different entities.

    entity :
        The particular entity whose flareness in the graph G we want to compute.

    weight :

    query_data : str
        The node attribute key containing 'unique members' (names of entities) of each node.

    verbose : bool
        whether or not to print diagnostic messages
    """


    G_entity = set(mfc.get_nodes_containing_entity(G, entity, query_data))
    core, shell = mfc.compute_core_shell(G, G_entity)

    if len(G_entity) == 0:
        if verbose > 0: print("Entity {} not found. Ignoring".format(entity))
        return None, None

    if verbose > 0:
        print("G_entity: ", G_entity)
        print("core: ", core)
        print("shell: ", shell)

    G_entity_subgraph = G.subgraph(G_entity)
    distances = {}
    if len(shell) > 0:
        distances = nx.multi_source_dijkstra_path_length(G_entity_subgraph,shell,
                                                         weight=weight)
    k = []
    components = list(nx.connected_components(G.subgraph(core)))
    for L in components:
        k_L = max(distances.get(x,np.inf) for x in L)
        k.append(k_L)

    return (k, components)




def compute_all_summary(G, entities, weight=(lambda v,u,e: 1),
                        query_data='unique_members', verbose=0,
                        keep_missing=False):
    """
    Compute "flareness" of all entities in Mapper graph G
    using the proposed definition in
    Escolar et al., "Mapping Firms' Locations in Technological Space"

    See mappertools.features.core.get_nodes_containing_entity for a discussion on entities.

    Parameters
    ----------
    G : networkx graph
        Represents a Mapper graph, where each node may be
        a set of observations of different entities.

    entities :
        List of entities whose flareness in the graph G we want to compute.

    weight :

    query_data : str
        The node attribute key containing 'unique members' (names of entities) of each node.

    verbose : bool
        whether or not to print diagnostic messages

    keep_missing : bool
        whether or not to include entities not found in G

    Returns
    -------
    ans : pandas.DataFrame
        With columns 'flare_type','flare_index','flare_sig', corresponding to
        type, flare length, and flare signature respectively.
        See the paper for definitions.
    """

    ans = pandas.DataFrame(columns=['flare_type','flare_index','flare_sig'])
    for entity in entities:
        k, _ = compute_flareness(G, entity, weight, query_data, verbose)
        if k is None:
            if keep_missing:
                ans.loc[entity] = pandas.Series({'flare_type':-1, 'flare_index':None, 'flare_sig':None})
            continue
        else:
            k_type, k_index = flare_type_index(k)
            ans.loc[entity] = pandas.Series({'flare_type':k_type, 'flare_index':k_index, 'flare_sig':k})

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
