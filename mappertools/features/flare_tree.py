import networkx as nx
import numpy as np
import operator

import itertools


class Flare(object):
    def __init__(self, node, birth):
        self.nodes = set([node])

        self.birth = (birth, node)
        self.death = (None, None)

    def __str__(self):
        return "Flare, with birth {} and death {}, members: {}".format(self.birth, self.death, self.nodes)

    def __repr__(self):
        return self.__str__()

    def __contains__(self, node):
        return node in self.nodes

    

class FlareTree:
    def __init__(self, flare, children=None, parent=None):
        self.flare = flare
        if children is None: self.children = set([])
        self.parent = parent

    def __iter__(self):
        # https://stackoverflow.com/a/6915269
        yield self
        for subtree in itertools.chain(*map(iter, self.children)):
            yield subtree

    def __contains__(self, item):
        for subtree in self:
            if item in subtree.flare: return True
        return False

    def intersects(self, items):
        for item in items:
            if item in self: return True
        return False

    def add_subtree(self, tree):
        if tree.parent is not None:
            raise RuntimeWarning("Cannot add subtree that already has a parent. Ignoring")
        else:
            tree.parent = self
            self.children.add(tree)

    def print_all(self):
        for subtree in self:
            print(subtree.flare)


def unpack_flares(flare_trees):
    return [subtree.flare for tree in flare_trees for subtree in tree]


def flare_detect(G, centrality):
    if isinstance(centrality, str): centrality = G.nodes.data(centrality)

    flare_trees = set([])
    for node, cur_cen in sorted(centrality.items(), key=operator.itemgetter(1)):
        print("node", node)
        neighbors = [nbr for nbr in G.adj[node] if centrality[nbr] <= cur_cen]
        print("nbhrs", neighbors)
        death_candidates = [tree for tree in flare_trees if tree.intersects(neighbors)]

        if len(death_candidates) == 0:
            print("birth")
            new_tree = FlareTree(flare=Flare(node, cur_cen))
            flare_trees.add(new_tree)
        else:
            elder_tree = min(death_candidates, key=(lambda tree: tree.flare.birth[0]))
            for tree in death_candidates:
                if tree != elder_tree:
                    flare_trees.remove(tree)
                    tree.flare.death = (cur_cen, node)
                    elder_tree.add_subtree(tree)
            elder_tree.flare.nodes.add(node)
        for tree in flare_trees:
            tree.print_all()
        print()

    return unpack_flares(flare_trees)


if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(0,-1),(-1,-2),(0,-3)])
    cen = nx.centrality.harmonic_centrality(G)
    print(cen)
    flare_trees = flare_detect(G, cen)

    for tree in flare_trees:
        for subtree in tree:
            print(subtree.flare)
