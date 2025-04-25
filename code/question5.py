import networkx as nx
import random
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, mean_absolute_error
import os

def load_graph(file_path):
    G = nx.read_gml(file_path)
    return G

def prepare_data(G, attribute, missing_fraction):
    labels = {str(node): data.get(attribute, None) for node, data in G.nodes(data=True)}
    # Remove nodes that don’t have this attribute
    labels = {k: v for k, v in labels.items() if v is not None}

    nodes = list(labels.keys())
    random.shuffle(nodes)
    num_missing = int(missing_fraction * len(nodes))
    unlabeled_nodes = nodes[:num_missing]
    labeled_nodes = nodes[num_missing:]

    y = {}
    for node in labeled_nodes:
        y[node] = labels[node]
    for node in unlabeled_nodes:
        y[node] = None

    return y, labels

def label_propagation(G, y, max_iter=100):
    y_pred = y.copy()
    for _ in range(max_iter):
        updated = False
        for node in G.nodes():
            node = str(node)
            if y_pred.get(node) is None:
                neighbor_labels = [y_pred.get(str(n)) for n in G.neighbors(node) if y_pred.get(str(n)) is not None]
                if neighbor_labels:
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    y_pred[node] = most_common
                    updated = True
        if not updated:
            break
    return y_pred

def evaluate(y_true, y_pred):
    y_true_list = []
    y_pred_list = []
    for node in y_true:
        if y_true[node] is not None and y_pred.get(node) is not None:
            y_true_list.append(y_true[node])
            y_pred_list.append(y_pred[node])
    
    if len(y_true_list) == 0:  # ⚠️ Avoid empty list crash
        return 0.0, float('nan')  # or return None, None

    accuracy = accuracy_score(y_true_list, y_pred_list)
    mae = mean_absolute_error(y_true_list, y_pred_list)
    return accuracy, mae

def run_experiment(file_path, attributes, missing_fractions):
    G = load_graph(file_path)
    results = {}
    for attribute in attributes:
        results[attribute] = {}
        for fraction in missing_fractions:
            y, y_true = prepare_data(G, attribute, fraction)
            y_pred = label_propagation(G, y)
            accuracy, mae = evaluate(y_true, y_pred)
            results[attribute][fraction] = {'accuracy': accuracy, 'mae': mae}
            print(f"Attribute: {attribute}, Missing Fraction: {fraction}")
            print(f"Accuracy: {accuracy:.4f}, MAE: {mae:.4f}\n")
    return results

if __name__ == "__main__":
    # Example on MIT dataset
    file_path = "data/data/MIT8.gml" 
    attributes = ['dorm', 'major_index', 'gender']
    missing_fractions = [0.1, 0.2, 0.3]
    run_experiment(file_path, attributes, missing_fractions)
