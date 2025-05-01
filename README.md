# Facebook100 Network Analysis Project

This project analyses social network properties using the Facebook100 dataset. It explores graph structures such as degree distributions, clustering, assortativity, link prediction, label propagation, and community detection.

## 🔍 Overview

The Facebook100 dataset contains friendship graphs of 100 U.S. universities on Facebook as of 2005. Each graph is stored in `.gml` format and contains node-level attributes like gender, major, dorm, degree, and student/faculty status.

This repository includes code and figures corresponding to a multi-part analysis assignment covering core topics in network science.

---

## 📂 Project Structure

```bash
.
├── code/                        # Python scripts for each question
│   ├── question2.py            # Degree distribution & clustering analysis
│   ├── question3.py            # Assortativity analysis (5 attributes)
│   ├── question4.py            # Link prediction (Common Neighbors, Jaccard, Adamic/Adar)
│   ├── question5.py            # Label propagation algorithm for attribute prediction
│   └── question6.py            # Community detection + hypothesis testing with Louvain
│
├── figures/                    # All plots and result images
│   ├── q2/                     # Degree distribution & clustering plots
│   ├── q3/                     # Assortativity scatter plots & histograms
│   ├── q4/                     # Precision, Recall, Top@k plots for link prediction
│   ├── q5/                     # Label propagation accuracy results
│   └── q6/                     # Community detection visualizations and NMI scores
│
├── data/                       # Directory for Facebook100 .gml graphs (not uploaded)
│   └── *.gml                   # Each file is a university friendship graph
│
└── README.md                   # This file
```

---

## 🚀 Questions Answered

### ✅ Question 2 – Degree Distribution & Clustering
- Plots degree distributions for Caltech, MIT, and Johns Hopkins.
- Computes global and local clustering coefficients and edge density.
- Analyse the relationship between node degree and clustering coefficient.

### ✅ Question 3 – Assortativity Analysis
- Computes attribute assortativity for:
  - Gender
  - Dorm
  - Degree
  - Major (via `major_index`)
  - Student/Faculty Status (via `student_fac`)
- Visualises results using histograms and scatter plots over network size.

### ✅ Question 4 – Link Prediction
- Implements 3 link prediction metrics from scratch:
  - Common Neighbours
  - Jaccard Coefficient
  - Adamic/Adar
- Evaluates using Precision@k, Recall@k, and Top@k.
- Run on 12 smallest graphs to balance coverage and efficiency.

### ✅ Question 5 – Label Propagation
- Uses Zhu et al.’s algorithm to predict missing labels.
- Tested on `MIT8.gml` for attributes: gender, major_index, dorm.
- Reports accuracy and mean absolute error at multiple missing data rates.

### ✅ Question 6 – Community Detection
- Hypothesis: dorm affiliation correlates with social groups.
- Uses Louvain algorithm to detect communities.
- Measures alignment with real dorm labels using Normalised Mutual Information (NMI).

---

## 📦 Requirements

Install required libraries using:

```bash
pip install networkx matplotlib numpy scikit-learn tqdm
```
---

## 🧠 Authors

Developed by Achref LOUSSAIEF & Marwen MBARKI
Course: Network Analysis — Réseaux complexes et validation (2025 - TSP ) 
Instructed by: Dr. Vincent Gauthier


