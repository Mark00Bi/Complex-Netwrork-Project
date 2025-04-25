import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# Create folder to save plots
os.makedirs("figures/q3", exist_ok=True)

# Attributes to study
attributes = ["student_fac", "major_index", "gender", "dorm", "degree"]

# Prepare storage
assortativity_results = {attr: [] for attr in attributes}
network_sizes = []

# Load and analyze each graph
data_dir = "data/data"
for filename in tqdm(sorted(os.listdir(data_dir))):
    if not filename.endswith(".gml"):
        continue

    filepath = os.path.join(data_dir, filename)
    try:
        G = nx.read_gml(filepath, label='id')
        G = G.to_undirected()
    except Exception as e:
        print(f"Could not read {filename}: {e}")
        continue

    n = G.number_of_nodes()
    network_sizes.append(n)

    # Degree assortativity
    assortativity_results["degree"].append(nx.degree_assortativity_coefficient(G))

    for attr in ["student_fac", "major_index", "gender", "dorm"]:
        try:
            assort = nx.attribute_assortativity_coefficient(G, attr)
        except Exception as e:
            assort = np.nan
        assortativity_results[attr].append(assort)

# Plotting
for attr in attributes:
    values = assortativity_results[attr]
    sizes = np.array(network_sizes)

    # Scatter plot
    plt.figure()
    plt.scatter(sizes, values, alpha=0.6)
    plt.xscale("log")
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Network Size (log scale)")
    plt.ylabel(f"{attr.capitalize()} Assortativity")
    plt.title(f"{attr.capitalize()} Assortativity vs Network Size")
    plt.grid(True)
    plt.savefig(f"figures/q3/{attr}_assortativity_vs_size.png")
    plt.close()

    # Histogram
    plt.figure()
    clean_vals = [v for v in values if not np.isnan(v)]
    plt.hist(clean_vals, bins=30, alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel(f"{attr.capitalize()} Assortativity")
    plt.ylabel("Frequency")
    plt.title(f"{attr.capitalize()} Assortativity Distribution")
    plt.grid(True)
    plt.savefig(f"figures/q3/{attr}_assortativity_histogram.png")
    plt.close()
