import networkx as nx
import numpy as np
import operator

import itertools


def flare_detect(G, centrality, prune_threshold=0, verbose=False):
    """
    Compute "flares" in G using the 0-persistent homology of centrality filtration.

    Parameters
    ----------
    G : networkx graph

    centrality : dict {node : centrality}, or
                 str  node attribute corresponding to centrality

    prune_threshold : float
        Flares with persistence lifespan less than prune_threshold
        are combined with parent trees.

    verbose : boolean
    """

    if isinstance(centrality, str): centrality = G.nodes.data(centrality)

    flare_trees = set([])
    for node, cur_cen in sorted(centrality.items(), key=operator.itemgetter(1)):
        neighbors = [nbr for nbr in G[node] if centrality[nbr] <= cur_cen]

        if verbose:
            print("node: {}, centrality: {}, neighbors: {}".format(node,
                                                                   cur_cen,
                                                                   neighbors))

        death_candidates = [tree for tree in flare_trees if tree.intersects(neighbors)]
        if len(death_candidates) == 0:
            # print("birth")
            new_tree = FlareTree(flare=Flare(node, cur_cen))
            flare_trees.add(new_tree)
        else:
            elder_tree = min(death_candidates, key=(lambda tree: tree.flare.birth[0]))
            for tree in death_candidates:
                if tree != elder_tree:
                    flare_trees.remove(tree)
                    tree.flare.death = (cur_cen, node)
                    elder_tree.add_subtree(tree)
                    if tree.flare.lifespan() < prune_threshold:
                        elder_tree.collapse_subtree(tree)

            elder_tree.flare.nodes.add(node)

        # for tree in flare_trees:
        #     tree.print_all()
        # print()
    ans = sort_flares(unpack_flares(flare_trees))
    return ans



class Flare(object):
    def __init__(self, node, birth):
        self.nodes = set([node])

        self.birth = (birth, node)
        self.death = (np.inf, None)

    def __str__(self):
        return "Flare, with birth {} and death {}, members: {}".format(self.birth, self.death, self.nodes)

    def __repr__(self):
        return self.__str__()

    def __contains__(self, node):
        return node in self.nodes

    def lifespan(self):
        return self.death[0] - self.birth[0]
    

class FlareTree:
    def __init__(self, flare, children=None, parent=None):
        self.flare = flare
        if children is None: self.children = set([])
        self.parent = parent

    def __iter__(self):
        """
        iterates through self and all subtrees (not just direct children)
        """
        # https://stackoverflow.com/a/6915269
        yield self
        for subtree in itertools.chain(*map(iter, self.children)):
            yield subtree

    def __contains__(self, item):
        """
        checks whether or not item is contained in self or any subtree
        """
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

    def collapse_subtree(self, tree):
        if tree not in self.children or not (self is tree.parent):
            raise RuntimeWarning("Cannot collapse tree, not valid child. Ignoring")
        else:
            tree_nodes = set().union(*(subtree.flare.nodes for subtree in tree))
            self.flare.nodes = self.flare.nodes.union(tree_nodes)

            tree.parent = None
            self.children.remove(tree)


    def print_all(self):
        for subtree in self:
            print(subtree.flare)


def unpack_flares(flare_trees):
    return [subtree.flare for tree in flare_trees for subtree in tree]

def sort_flares(flare_list):
    return sorted(flare_list, key=(lambda flare:flare.death[0] - flare.birth[0]),reverse=True)





if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(0,-1),(-1,-2),(0,-3)])
    cen = nx.centrality.harmonic_centrality(G)
    print(cen)
    flare_trees = flare_detect(G, cen)

    for tree in flare_trees:
        for subtree in tree:
            print(subtree.flare)
