import networkx as nx
import matplotlib.pyplot as plt
paths = []
path = []


def get_adj_dict(Graph: nx.Graph):
    adj_dict = dict()
    adj_iter = Graph.adjacency()
    for n, nbrdict in adj_iter:
        adj_dict[n] = list(dict(nbrdict).keys())
    return adj_dict


def dfs(start, curr, end, adj_dict):
    """ 用递归的方式寻找起点到终点的所有路径
    """
    path.append(curr)
    if curr == end:
        paths.append(path.copy())
        path.pop()
    else:
        for item in adj_dict[curr]:
            if item not in path:
                dfs(start, item, end, adj_dict)
        path.pop()


def lp(Graph: nx.Graph, cut_pairs: list):
    """ 线性优化器, 有时候会发出警告, 但是不影响结果
    """
    adj_dict = get_adj_dict(Graph)
    edges = list(Graph.edges())
    edges_with_weight = list()
    import numpy as np
    c = np.ones(len(edges))
    A_ub = np.zeros((0, len(edges)))
    for cut_pair in cut_pairs:
        start = cut_pair[0]
        end = cut_pair[1]
        del paths[:]
        del path[:]
        dfs(start, start, end, adj_dict)
        for p in paths:
            A_ub_row = np.zeros((1, len(edges)))
            for i in range(len(p)-1):
                one_edge = (p[i], p[i+1])
                try:
                    index = edges.index(one_edge)
                except Exception:
                    one_edge = (p[i+1], p[i])
                    index = edges.index(one_edge)
                A_ub_row[0][index] = -1
            A_ub = np.concatenate((A_ub, A_ub_row))
    rows_n = A_ub.shape[0]
    b_ub = -1 * np.ones(rows_n)
    bounds = (0, None)
    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    dist = res.x
    Graph_with_dist = nx.Graph()
    Graph_with_dist.add_nodes_from(list(Graph.nodes()))
    for ((u, v), weight) in zip(edges, dist):
        edge_with_weight = (u, v, weight)
        edges_with_weight.append(edge_with_weight)
    Graph_with_dist.add_weighted_edges_from(edges_with_weight)
    return Graph_with_dist


def multicut(Graph: nx.Graph, cut_pairs, sources: list):
    Graph_with_dist = lp(Graph, cut_pairs)
    edges = set(Graph.edges())
    cut_edges = set()
    import random
    r = random.random()/2
    order = list(range(1, len(sources)+1))
    import itertools
    permutations = list(itertools.permutations(order))
    permutation = random.choice(permutations)
    for i in permutation:
        source = sources[i-1]
        capture_edges = set()
        for edge in list(edges):
            u = edge[0]
            v = edge[1]
            dist_u = nx.dijkstra_path_length(Graph_with_dist, source, u)
            dist_v = nx.dijkstra_path_length(Graph_with_dist, source, v)
            if (dist_u <= r) ^ (dist_v <= r):
                cut_edges.add(edge)
            if (dist_u <= r) and (dist_v <= r):
                capture_edges.add(edge)
        edges.difference_update(cut_edges)
        edges.difference_update(capture_edges)
    return list(cut_edges)


def multicut_solver(Graph: nx.Graph, terminal_lists: list):
    """ terminal_lists 是列表的列表, 其中每一个元素都是一些点的列表
    表示这些点不能连通
    """
    import itertools
    import random
    cut_pairs = []
    sources = []
    for terminal_list in terminal_lists:
        permutations = list(itertools.permutations(terminal_list))
        terminal_list = random.choice(permutations)
        sources.extend(terminal_list[:-1])
        cut_pairs.extend(list(itertools.combinations(terminal_list, 2)))
    return multicut(Graph, cut_pairs, sources)


Graph = nx.Graph()
Graph.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Graph.add_edges_from([
    (1, 2), (1, 3), (1, 7), (2, 3), (2, 4),
    (3, 8), (7, 8), (2, 9), (5, 10), (4, 10),
    (3, 4), (3, 6), (4, 5), (5, 6), (6, 7)]
)
nx.draw(Graph, with_labels=True)
plt.show()
cut = list(Graph.edges())
for i in range(20):
    solution_cut = multicut_solver(Graph, [[1, 4], [2, 5, 7]])
    if len(solution_cut) < len(cut):
        cut = solution_cut
# print(cut)
print(f"An Approximate cut's size is {len(cut)}.")
Graph.remove_edges_from(cut)
nx.draw(Graph, with_labels=True)
plt.show()
