import networkx as nx

G = nx.path_graph(4)

centrality = nx.eigenvector_centrality_numpy(G)

print("nodes：", list(G.nodes))
print("edge：", list(G.edges))
print("val in centrality：")
for node, val in centrality.items():
    print(f"node {node}: {val:.4f}")
