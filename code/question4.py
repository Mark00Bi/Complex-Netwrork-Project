from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
from itertools import combinations
import math

class LinkPrediction(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.N = len(graph)

    def neighbors(self, v):
        return list(self.graph.neighbors(v))

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")


class CommonNeighbors(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self):
        scores = dict()
        for u, v in combinations(self.graph.nodes(), 2):
            if not self.graph.has_edge(u, v):
                common = len(set(self.neighbors(u)) & set(self.neighbors(v)))
                if common > 0:
                    scores[(u, v)] = common
        return scores


class Jaccard(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self):
        scores = dict()
        for u, v in combinations(self.graph.nodes(), 2):
            if not self.graph.has_edge(u, v):
                Nu = set(self.neighbors(u))
                Nv = set(self.neighbors(v))
                union = Nu | Nv
                inter = Nu & Nv
                if len(union) > 0:
                    scores[(u, v)] = len(inter) / len(union)
        return scores


class AdamicAdar(LinkPrediction):
    def __init__(self, graph):
        super().__init__(graph)

    def fit(self):
        scores = dict()
        for u, v in combinations(self.graph.nodes(), 2):
            if not self.graph.has_edge(u, v):
                common = set(self.neighbors(u)) & set(self.neighbors(v))
                score = sum(1 / math.log(len(self.neighbors(z))) for z in common if len(self.neighbors(z)) > 1)
                if score > 0:
                    scores[(u, v)] = score
        return scores
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import os

def evaluate_predictions(true_edges, predicted_scores, k_vals=[50, 100, 200, 400]):
    true_set = set(true_edges)
    predicted_sorted = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = defaultdict(list)

    for k in k_vals:
        top_k = [e[0] for e in predicted_sorted[:k]]
        tp = len(set(top_k) & true_set)
        fp = k - tp
        fn = len(true_set) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        topk = tp / k

        results['k'].append(k)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['topk'].append(topk)

    return results


def link_prediction_experiment(graph_path, fraction=0.1, k_vals=[50, 100, 200, 400]):
    G = nx.read_gml(graph_path, label='id')
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    edges = list(G.edges())
    num_remove = int(fraction * len(edges))
    removed = random.sample(edges, num_remove)
    G.remove_edges_from(removed)

    predictors = {
        "CommonNeighbors": CommonNeighbors(G),
        "Jaccard": Jaccard(G),
        "AdamicAdar": AdamicAdar(G)
    }

    results_all = {}
    for name, predictor in predictors.items():
        print(f"Running {name}...")
        scores = predictor.fit()
        results = evaluate_predictions(removed, scores, k_vals)
        results_all[name] = results

    return results_all


def plot_results(results_all, output_prefix="results"):
    os.makedirs("figures/q4", exist_ok=True)
    for method, results in results_all.items():
        print(f"{method} Precision@k: {results['precision']}")

        # Precision@k plot
        plt.figure()
        plt.plot(results['k'], results['precision'], marker='o', color='blue')
        plt.xlabel("k")
        plt.ylabel("Precision@k")
        plt.title(f"{method} - Precision@k")
        plt.grid(True)
        plt.savefig(f"figures/q4/{output_prefix}_{method}_precision.png")
        plt.close()

        # Recall@k plot
        plt.figure()
        plt.plot(results['k'], results['recall'], marker='s', color='orange')
        plt.xlabel("k")
        plt.ylabel("Recall@k")
        plt.title(f"{method} - Recall@k")
        plt.grid(True)
        plt.savefig(f"figures/q4/{output_prefix}_{method}_recall.png")
        plt.close()

        # Top@k plot
        plt.figure()
        plt.plot(results['k'], results['topk'], marker='^', color='green')
        plt.xlabel("k")
        plt.ylabel("Top@k")
        plt.title(f"{method} - Top@k")
        plt.grid(True)
        plt.savefig(f"figures/q4/{output_prefix}_{method}_topk.png")
        plt.close()

if __name__ == "__main__":
    graph_file = "data/data/Caltech36.gml"
    results = link_prediction_experiment(graph_file, fraction=0.1)
    plot_results(results, output_prefix="caltech")