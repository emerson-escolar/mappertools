import numpy
import pandas
import copy
import networkx as nx

import json
import kmapper as km

import mappertools.features.flare_tree as flr


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


def nxmapper_append_flare_numbers(nxgraph):

    choices = ((nx.centrality.harmonic_centrality, "H"),
               (nx.centrality.closeness_centrality, "C"),
               (nx.centrality.betweenness_centrality, "B")
               )

    for fun, code in choices:
        centrality = fun(nxgraph)
        flares = flr.flare_detect(nxgraph, centrality)

        label = code + 'flare'
        for idx, flare in enumerate(flares):
            for node in flare.nodes:
                nxgraph.nodes[node][label] = idx

        label = code + 'centrality'
        for node, cen in centrality.items():
            nxgraph.nodes[node][label] = cen

    # long_flares = flr.threshold_flares(flares)
    # for idx, flare in enumerate(long_flares):
    #     for node in flare.nodes:
    #         nxgraph.nodes[node]['longflare'] = idx

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


def kmapper_to_nxmapper(graph, counts=True, weights=True):
    """
    Convenience function for turning kmapper output into networkx format with extra data

    Assumptions
    -----------
    graph is a kmapper output graph

    Parameters
    ----------
    counts : bool
        whether or not to append membership 'count' data to nodes and edges, and 'weight' data to edges.

    weights : bool
        whether or not to append membership 'weight' data to edges
    """

    nxgraph = km.adapter.to_nx(graph)

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

    return nxgraph

def cytoscapejson_dump(graph, file,
                       members_extra_data, edges_extra_data,
                       node_transforms=None, edge_transforms=None, compute_flares=False):
    """
    Dump kmapper graph, with extra data, as cytoscape json file.

    Assumptions
    -----------
    graph is a kmapper output graph

    Parameters
    ----------
    members_extra_data : dict of dicts {key : {index : data}}
        For each node, for each *key*, append list of *data* corresponding to its list of member *index*es

    node_transforms: dict {key : function}
        For each *key*, apply *function* to the extra_data appended to nodes

    edges_extra_data : dict of dicts {key : {index : data}}
        For each edge, for each *key*, append list of *data* corresponding to its list of member *index*es

    edge_transforms: dict {key : function}
        For each *key*, apply *function* to the extra_data appended to edges
    """

    nxGraph = kmapper_to_nxmapper(graph, counts=True, weights=True)
    nxGraph = nxmapper_append_node_member_data(nxGraph, members_extra_data, node_transforms)
    nxGraph = nxmapper_append_edge_member_data(nxGraph, edges_extra_data, edge_transforms)

    if compute_flares:
        nxGraph = nxmapper_append_flare_numbers(nxGraph)

    with open(file, 'w') as outfile:
       json.dump(nx.readwrite.json_graph.cytoscape_data(nxGraph), outfile)

    return nxGraph



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


def cytoscapejson_dump_with_averages(data, graph, file, averager=(lambda x: numpy.mean(x, axis=0))):
    nxGraph = km.adapter.to_nx(graph)

    aves = _compute_averages(data, graph, averager)

    for key, value_dicts in aves.to_dict('index').items():
        nx.set_node_attributes(nxGraph, value_dicts, key)

    with open(file, 'w') as outfile:
        json.dump(nx.readwrite.json_graph.cytoscape_data(nxGraph), outfile)

    return nxGraph





def kmapper_text_dump(graph, file, labels=None):
    print("Nodes", file=file)
    for node, members in graph['nodes'].items():
        print("#" + node, file=file)
        print(len(members), file=file)

        if not labels == None:
            for mem in members:
                print(labels[mem], file=file)
        else:
            print(members, file=file)


    print("Links", file=file)
    for cluster, links in graph['links'].items():
        print(cluster, file=file)
        print(links, file=file)

    if not labels == None:
        print("Labels", file=file)
        for idx, val in enumerate(labels):
            print(str(idx) + " " + str(val), file=file)


def kmapper_dump_cluster_averages(data, graph, file, averager=(lambda x: numpy.mean(x, axis=0))):
    aves = _compute_averages(data, graph, averager)
    aves.to_csv(file, index=True, sep='\t')


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
