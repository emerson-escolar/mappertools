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
        print(node, cur_cen)

        death_candidates = set()
        for nbr in G.adj[node]:
            if centrality[nbr] > cur_cen:
                continue
            for flare, idx in enumerate(flares):
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


if __name__ == "__main__":
    print("hi")

    flare_detect(None, {2:-1,3:2,1:-1})
