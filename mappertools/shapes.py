import networkx as nx
import itertools
import numpy as np
import sklearn.neighbors

## In development. DO NOT USE?

def mesh_to_nxgraph(mesh):
    """
    Converts pymesh mesh to 1-skeleton as networkx graph

    PyMesh (https://github.com/qnzhou/PyMesh, not the one on PyPI)
    """

    G = nx.Graph()
    for x in mesh.faces:
        for pr in itertools.combinations(x, 2):
            G.add_edge(*pr)
    for pr in G.edges:
        i,j=pr
        G[i][j]["weight"] = np.linalg.norm( mesh.vertices[i,:] - mesh.vertices[j,:])
    return G



def nxgraph_to_dijkstra_distance_matrix(G):
    N = len(G)
    if nx.number_connected_components(G) != 1:
        print("Warning: Graph is not connected!")

    ans = np.full( (N,N), np.inf)

    dist = nx.all_pairs_dijkstra_path_length(G)
    for idx, data in dist:
        for jdx, w in data.items():
            ans[idx][jdx] = w

    return ans


# not very efficient:
class DistanceQuery(object):
    def __init__(self, dists, points):
        self.dists = dists
        self.points = points
        self.nn = sklearn.neighbors.NearestNeighbors(n_neighbors=1).fit(self.points)

    def __call__(self, x, y):
        d, idx = self.nn.kneighbors(np.stack((x,y),axis=0))
        ans= np.sum(d) + self.dists[idx[0][0]][idx[1][0]]
        return ans


def distance_matrix_to_function(dists, points):
    pass
