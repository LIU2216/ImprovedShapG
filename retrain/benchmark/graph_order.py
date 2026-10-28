"""Rank-deletion edge order (mirrors shapG's GraphBuilder.from_rank_deletion).

Nodes are processed in order of relevance to the target (ascending: least relevant first, set by
RANK_DESC=0 in the entry scripts); for each node its edges are tried by weight, strongest first, and
an edge is deleted unless that disconnects the graph. Adding the deleted edges back in reverse order
walks from the sparsest connected graph (a spanning tree) back to the complete graph.
"""
import networkx as nx
from shapG import GraphBuilder


def get_deletion_order(X, y, corr_method="cosine", sim_method="cosine", complete_graph=None):
    """Return (complete graph, sparsest connected graph, deleted edges as (u, v, w) in deletion order;
    the first deleted is added back last)."""
    features = X.columns.tolist()
    feature_rank_indices = GraphBuilder.get_feature_rank(X, y, corr_method)
    feature_rank = [features[i] for i in feature_rank_indices]

    if complete_graph is None:
        sim_matrix = GraphBuilder._calculate_similarity_matrix(X.to_numpy(), sim_method)
        complete_graph = nx.Graph()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                complete_graph.add_edge(features[i], features[j], weight=sim_matrix[i, j])

    G = complete_graph.copy()
    deleted_edges = []
    for node in feature_rank:
        for u, v in sorted(G.edges(node), key=lambda e: G[e[0]][e[1]]["weight"], reverse=True):
            w = G[u][v]["weight"]
            G.remove_edge(u, v)
            if nx.is_connected(G):
                deleted_edges.append((u, v, w))
            else:
                G.add_edge(u, v, weight=w)
    return complete_graph, G, deleted_edges
