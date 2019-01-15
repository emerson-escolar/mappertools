import numpy
import pandas
import copy
import networkx as nx

import json

import kmapper as km


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


def cytoscapejson_dump(graph, members_extra_data, file):
    nxGraph = km.adapter.to_nx(graph)

    for key, data_map in members_extra_data.items():
        for node, membership in nxGraph.nodes.data('membership'):
            nxGraph.nodes[node][key] = [data_map[k] for k in membership]

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
