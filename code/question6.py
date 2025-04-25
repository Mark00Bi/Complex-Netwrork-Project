import networkx as nx
import community as community_louvain 
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

def load_graph_with_dorms(path):
    G = nx.read_gml(path)
    G = nx.convert_node_labels_to_integers(G)
    dorm_labels = nx.get_node_attributes(G, 'dorm')
    dorm_labels = {int(k): v for k, v in dorm_labels.items() if v is not None}
    return G, dorm_labels

def run_louvain(G):
    partition = community_louvain.best_partition(G)
    return partition

def compute_nmi(partition, ground_truth):
    # Extract aligned label vectors
    nodes = list(set(partition.keys()) & set(ground_truth.keys()))
    pred_labels = [partition[n] for n in nodes]
    true_labels = [ground_truth[n] for n in nodes]
    return normalized_mutual_info_score(true_labels, pred_labels)

if __name__ == "__main__":
    files = ["data/data/Caltech36.gml", "data/data/MIT8.gml", "data/data/Harvard1.gml"]
    for path in files:
        print(f"\n=== Processing {path} ===")
        G, dorm_labels = load_graph_with_dorms(path)
        partition = run_louvain(G)
        nmi = compute_nmi(partition, dorm_labels)
        print(f"NMI between Louvain communities and dorm labels: {nmi:.4f}")