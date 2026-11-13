import networkx as nx

G_path = nx.path_graph(4)
cent_path = nx.eigenvector_centrality_numpy(G_path)
print("Eigenvector centrality:", cent_path)

G_cycle = nx.cycle_graph(4)
cent_cycle = nx.eigenvector_centrality_numpy(G_cycle)
print("Eigenvector centrality:", cent_cycle)
