import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

# Create the figures folder if it doesn't exist
os.makedirs("figures", exist_ok=True)

# List of networks to study
networks = {
    "Caltech": "data/data/Caltech36.gml",
    "MIT": "data/data/MIT8.gml",
    "JohnsHopkins": "data/data/Johns Hopkins55.gml"
}

# Function to load a Facebook100 graph from .gml
def load_graph(filepath):
    G = nx.read_gml(filepath, label='id')  # Use 'id' field for node labels
    return G

# Analyze each network
for name, path in networks.items():
    print(f"\n=== Analyzing {name} ===")
    G = load_graph(path)

    # 1. Plot Degree Distribution
    degrees = [d for n, d in G.degree()]
    plt.figure()
    plt.hist(degrees, bins=50, log=True)
    plt.title(f"{name} Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency (log)")
    plt.grid(True)
    plt.savefig(f"figures/{name}_degree_distribution.png")
    plt.close()

    # 2. Compute clustering coefficients and density
    global_clustering = nx.transitivity(G)
    mean_local_clustering = nx.average_clustering(G)
    density = nx.density(G)

    print(f"Global clustering coefficient: {global_clustering:.4f}")
    print(f"Mean local clustering coefficient: {mean_local_clustering:.4f}")
    print(f"Edge density: {density:.6f}")

    # 3. Degree vs Local Clustering Scatter Plot
    clustering = nx.clustering(G)
    node_degrees = np.array(degrees)
    node_clustering = np.array([clustering[node] for node in G.nodes()])

    plt.figure()
    plt.scatter(node_degrees, node_clustering, alpha=0.5)
    plt.xscale('log')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Local Clustering Coefficient')
    plt.title(f"{name}: Degree vs Clustering")
    plt.grid(True)
    plt.savefig(f"figures/{name}_degree_vs_clustering.png")
    plt.close()
