import heapq
import networkx as nx
import copy


def adjacent_edge_weight_sum(node, G, edge_weight_string = "weight"):
    ans = 0
    for _, _, w in G.edges(node, edge_weight_string):
        ans += w
    return ans

def graph_peeling(G, edge_weight_string = "weight"):
    adj_weight_dict = {node: adjacent_edge_weight_sum(node) for node in G.nodes()}

    heap = [(weight, node) for node, weight in adj_weight_dict]
    heapq.heapify(heap)

    S = set(G.nodes())
    in_weight = sum([w for _,_,w in G.edges.data(edge_weight_string)])
    cur_val = in_weight / len(S)
    cur_S = copy.deepcopy(S)

    while n > 1:
        adj_weight, min_v = heapq.heappop(heap)

        if min_v in S:
            S.remove(min_v)
            in_weight -= adj_weight
            val = in_weight / len(S)

            if val > cur_val:
                cur_val = val
                cur_S = copy.deepcopy(S)

            for v in G[min_v]:
                if v in S:
                    adj_weight_dict[v] -= G[min_v][v][edge_weight_string]
                    heapq.heappush(heap, (adj_weight_dict[v],v))
    return cur_val, cur_S
