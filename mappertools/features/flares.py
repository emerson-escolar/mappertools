import networkx as nx
import numpy as np
import operator


class Flare(object):
    def __init__(self, node, birth):
        self.nodes = set([node])

        self.birth = birth
        self.death = None

        self.originator = node
        self.terminator = None

        self.finished = False

    def terminate(self, d_node, d_val):
        self.death = d_val
        self.terminator = d_node
        self.finished = True

    def __str__(self):
        return "Flare, with birth {} and death {}, members: {}".format(self.birth, self.death, self.nodes)

    def __repr__(self):
        return self.__str__()



def find_elder_flare(flares, candidates):
    min_birth = np.inf
    min_idx = None

    for idx in candidates:
        if flares[idx].birth < min_birth:
            min_idx = idx
            min_birth = flares[idx].birth
    return min_idx


def flare_detect(G, centrality):
    if isinstance(centrality, str):
        centrality = G.nodes.data(centrality)

    flares = []
    for node, cur_cen in sorted(centrality.items(), key=operator.itemgetter(1)):
        death_candidates = set()
        for nbr in G.adj[node]:
            if centrality[nbr] > cur_cen:
                continue
            for idx, flare in enumerate(flares):
                if not flare.finished and nbr in flare.nodes:
                    death_candidates.add(idx)
        if len(death_candidates) == 0:
            flares.append(Flare(node, cur_cen))
        else:
            elder = find_elder_flare(flares, death_candidates)
            for idx in death_candidates:
                if idx != elder:
                    flares[idx].terminate(node, cur_cen)
            flares[elder].nodes.add(node)

    return flares


def percentile_gap_threshold(data, percentile):
    hist, bin_edges = np.histogram(data, bins='doane')
    threshold = -np.inf
    gaps = np.argwhere(hist==0).flatten()
    if len(gaps) != 0:
        idx = np.percentile(gaps, 0,interpolation='nearest')
        threshold = bin_edges[idx]
    return threshold


def threshold_flares(flares, threshold=None):
    b,d = compute_pd(flares)
    d[d==None] = np.max(d[d != None])

    lifespans = d-b
    if threshold == None:
        threshold = percentile_gap_threshold(lifespans,0)
    long_flares = []
    for idx,flare in enumerate(flares):
        if lifespans[idx] >= threshold:
            long_flares.append(flare)
    return long_flares


def compute_pd(flares):
    b = np.array([flare.birth for flare in flares])
    d = np.array([flare.death for flare in flares])

    return b,d

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(0,-1),(-1,-2),(-1,1)])

    print(flare_detect(G, nx.centrality.harmonic_centrality(G)))
