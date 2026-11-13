import networkx as nx

# create an undirected graph
G = nx.Graph()

# add nodes
G.add_nodes_from(["A", "B", "C", "D"])
# add edges
G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")])

print("All nodes:", list(G.nodes))
print("All edges:", list(G.edges))
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
# calculate degree of each node
print("Node degree:")
for node, degree in G.degree():
    print(f"  {node}: {degree}")