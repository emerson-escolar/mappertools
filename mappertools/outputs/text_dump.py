import numpy
import pandas
import copy
import networkx as nx

import json
import kmapper as km

import mappertools.features.flare_tree as flare_tree



### ******************** networkx-based outputs ********************

def nxmapper_append_node_member_data(nxgraph, extra_data, transforms=None):
    """
    Assumptions
    -----------
    Each node in nxgraph has 'membership' data that contains indices of original observations.

    Parameters
    ----------
    extra_data : dict of dicts {key : {index : data}}
        For each node, for each *key*, append list of *data* corresponding to its list of member *index*es

    transforms: dict {key : function}
        For each *key*, apply *function* to the extra_data appended to nodes
    """
    for key, data_map in extra_data.items():
        if transforms == None or key not in transforms:
            fun = (lambda x:x)
        else:
            fun = transforms[key]
        for node, membership in nxgraph.nodes.data('membership'):
            nxgraph.nodes[node][key] = fun([data_map[k] for k in membership])

    return nxgraph


def nxmapper_append_centrality_flare_numbers(nxgraph):

    choices = ((nx.centrality.harmonic_centrality, "H"),
               (nx.centrality.closeness_centrality, "C"),
               (nx.centrality.betweenness_centrality, "B")
               )

    for fun, code in choices:
        centrality = fun(nxgraph)
        flares = flare_tree.flare_detect(nxgraph, centrality, prune_threshold=0.01)

        label = code + 'flare'
        for idx, flare in enumerate(flares):
            for node in flare.nodes:
                nxgraph.nodes[node][label] = idx

        label = code + 'centrality'
        for node, cen in centrality.items():
            nxgraph.nodes[node][label] = cen

    return nxgraph



def nxmapper_append_edge_member_data(nxgraph, extra_data, transforms=None):
    """
    Assumptions
    -----------
    Each edge in nxgraph has 'membership' data which is a list of indices of original observations.

    Parameters
    ----------
    extra_data : dict of dicts {key : {index : data}}
        For each edge, for each *key*, append list of *data* corresponding to its list of member *index*es

    transforms: dict {key : function}
        For each *key*, apply *function* to the extra_data appended to edges
    """
    for key, data_map in extra_data.items():
        if transforms == None or key not in transforms:
            fun = (lambda x:x)
        else:
            fun = transforms[key]
        for u, v, membership in nxgraph.edges.data('membership'):
            nxgraph.edges[(u,v)][key] = fun([data_map[k] for k in membership])

    return nxgraph


def nxmapper_append_basic_data(nxgraph, counts=True, weights=True, cen_flares=False):
    """
    Convenience function for appending networkx format mapper graph with counts and weights

    Assumption
    ----------
    Each node in nxgraph has 'membership' data that contains indices of original observations.

    Parameters
    ----------
    counts : bool
        whether or not to append membership 'count' data to nodes and edges, and 'weight' data to edges.

    weights : bool
        whether or not to append membership 'weight' data to edges

    cen_flares : bool
        whether or not to append 'flare index' data to nodes,
        as computed by persistence-based algorithm on centrality filtration
    """


    # Append edge membership
    for u,v in nxgraph.edges:
        u_mem = set(nxgraph.nodes[u]['membership'])
        v_mem = set(nxgraph.nodes[v]['membership'])
        nxgraph.edges[(u,v)]['membership'] =  list(u_mem.intersection(v_mem))

        if counts:
            nxgraph.edges[(u,v)]['count'] =  len(nxgraph.edges[(u,v)]['membership'])
        if weights:
            nxgraph.edges[(u,v)]['weight'] =  nxgraph.edges[(u,v)]['count'] / len(list(u_mem.union(v_mem)))

    # Append node membership counts
    if counts:
        for node, membership in nxgraph.nodes.data('membership'):
            nxgraph.nodes[node]["count"] = len(membership)

    if cen_flares:
        nxGraph = nxmapper_append_centrality_flare_numbers(nxgraph)

    return nxgraph



def kmapper_to_nxmapper(graph,
                        node_extra_data=None, edge_extra_data=None,
                        node_transforms=None, edge_transforms=None,
                        counts = True, weights = True, cen_flares=False):
    """
    Convenience function for converting kmapper graph to networkx format
    and appending extra data.

    Assumptions
    -----------
    graph is a kmapper output graph

    Parameters
    ----------
    node_extra_data : dict of dicts {key : {index : data}}
        For each node, for each *key*, append list of *data* corresponding to its list of member *index*es

    node_transforms: dict {key : function}
        For each *key*, apply *function* to the extra_data appended to nodes

    edge_extra_data : dict of dicts {key : {index : data}}
        For each edge, for each *key*, append list of *data* corresponding to its list of member *index*es

    edge_transforms: dict {key : function}
        For each *key*, apply *function* to the extra_data appended to edges

    counts : bool
        whether or not to append membership 'count' data to nodes and edges, and 'weight' data to edges.

    weights : bool
        whether or not to append membership 'weight' data to edges

    cen_flares : bool
        whether or not to append 'flare index' data to nodes,
        as computed by persistence-based algorithm on centrality filtration
    """

    nxGraph = km.adapter.to_nx(graph)
    nxGraph = nxmapper_append_basic_data(nxGraph, counts, weights, cen_flares)
    if node_extra_data:
        nxGraph = nxmapper_append_node_member_data(nxGraph, node_extra_data, node_transforms)
    if edge_extra_data:
        nxGraph = nxmapper_append_edge_member_data(nxGraph, edge_extra_data, edge_transforms)

    return nxGraph


### ******************** cytoscape json outputs ********************

def cytoscapejson_dump(nxgraph, file, positions=None):
    result_json = nx.readwrite.json_graph.cytoscape_data(nxgraph)
    if positions is not None:
        N = len(result_json["elements"]["nodes"])
        for i in range(N):
            result_json["elements"]["nodes"][i]["position"] = {"x": positions[i, 0], "y": positions[i, 1]}

    with open(file, 'w') as outfile:
       json.dump(result_json, outfile)

    return nxgraph



### ******************** text file outputs ********************

def kmapper_text_dump(graph, outfile, labels=None):
    print("Nodes", file=outfile)
    for node, members in graph['nodes'].items():
        print("#" + node, file=outfile)
        print(len(members), file=outfile)

        if not labels == None:
            for mem in members:
                print(labels[mem], file=outfile)
        else:
            print(members, file=outfile)


    print("Links", file=outfile)
    for cluster, links in graph['links'].items():
        print(cluster, file=outfile)
        print(links, file=outfile)

    if not labels == None:
        print("Labels", file=outfile)
        for idx, val in enumerate(labels):
            print(str(idx) + " " + str(val), file=outfile)


def _compute_averages(data, graph, averager=(lambda x: numpy.mean(x, axis=0))):
    index = list(graph['nodes'].keys())
    columns = list(data.columns)
    temp = numpy.zeros( (len(index), len(columns)) )
    for i in range(len(index)):
        node = index[i]
        members = graph['nodes'][node]
        temp[i,:] = averager(data.values[members,:])
    ans = pandas.DataFrame(temp, index=index)
    ans.columns=columns

    return ans.T


def kmapper_dump_cluster_averages(data, graph, outfile,
                                  averager=(lambda x: numpy.mean(x, axis=0))):
    aves = _compute_averages(data, graph, averager)
    aves.to_csv(outfile, index=True, sep='\t')


### ******************** utilities ********************

def kmapper_clean_graph(graph, min_num_members=2):
    ans = copy.deepcopy(graph)
    deleted_clusters = set()

    for node in graph['nodes'].keys():
        if not(len(graph['nodes'][node]) >= min_num_members):
            del (ans['nodes'])[node]
            deleted_clusters.add(node)

    for node in graph['links'].keys():
        if node in deleted_clusters:
            del (ans['links'])[node]
        else:
            ans['links'][node] = list(set(ans['links'][node]).difference(deleted_clusters))

    return ans
