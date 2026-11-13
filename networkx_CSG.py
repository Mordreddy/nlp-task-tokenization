import networkx as nx

G = nx.path_graph(4)
print("nodes:", list(G.nodes))
print("edges:", list(G.edges))
