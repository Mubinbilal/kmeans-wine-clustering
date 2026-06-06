# kmeans-wine-clustering
K-Means Clustering on the Wine dataset — uses the Elbow Method to find optimal k, applies StandardScaler for feature normalization, visualizes clusters with centroids, and evaluates quality using Silhouette Score.
# 🍷 K-Means Clustering on Wine Dataset

A clean implementation of **K-Means Clustering** using Python and Scikit-learn. This project applies unsupervised learning to the Wine dataset, uses the **Elbow Method** to determine the optimal number of clusters, visualizes cluster assignments with centroids, and evaluates cluster quality using the **Silhouette Score**.

---

## 🔍 Overview

This notebook walks through a complete K-Means clustering workflow:

1. **Load & Explore** the Wine dataset — shape, features, first 5 rows
2. **Select** two key features — `alcohol` and `malic_acid` — for 2D clustering
3. **Scale** features using StandardScaler (essential for distance-based clustering)
4. **Find Optimal K** using the Elbow Method — testing k from 1 to 10
5. **Train** K-Means with k=3 clusters and print cluster centroids
6. **Visualize** cluster assignments and centroids on a 2D scatter plot
7. **Evaluate** cluster quality with the Silhouette Score
8. **Analyze** sample distribution across all 3 clusters

---

## 📁 Project Structure

```
kmeans-wine-clustering/
│
├── k_means_clustering_implementation.ipynb   # Main Jupyter Notebook
└── README.md                                 # Project documentation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| NumPy | Array operations |
| Pandas | Dataset loading and DataFrame manipulation |
| Scikit-learn | Dataset, KMeans, StandardScaler, Silhouette Score |
| Matplotlib | Elbow curve and cluster scatter plot visualization |
| Jupyter Notebook | Interactive development environment |

---

## 📊 Dataset — Wine

The built-in **Wine dataset** from Scikit-learn:

| Property | Value |
|---|---|
| Total Samples | 178 |
| Total Features | 13 (chemical measurements) |
| Classes (Ground Truth) | 3 wine cultivars |
| Features Used | `alcohol`, `malic_acid` |

> K-Means is **unsupervised** — class labels are not used during training. The algorithm discovers natural groupings from the feature data alone.

---

## ⚙️ Model Configuration

```python
# Feature Scaling — mandatory for distance-based clustering
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_selected)

# K-Means Model
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

- **`n_clusters=3`** — chosen from the Elbow Method analysis
- **`n_init=10`** — runs 10 different centroid initializations, picks the best
- **`random_state=42`** — ensures reproducible results
- **`StandardScaler`** — normalizes features to zero mean and unit variance; critical since KMeans uses Euclidean distance

---

## 📈 Elbow Method — Finding Optimal K

The Elbow Method plots inertia (within-cluster sum of squares) for k = 1 to 10. The optimal k is where the curve bends — forming an "elbow":

```
k=1 → high inertia (one big cluster)
k=3 → elbow point ✅ — significant drop, chosen as optimal
k=10 → diminishing returns
```

---

## 📉 Evaluation — Silhouette Score

The **Silhouette Score** measures how well each sample fits its own cluster vs. neighboring clusters:

| Score Range | Interpretation |
|---|---|
| Close to +1.0 | Well-separated, dense clusters |
| Around 0.0 | Overlapping clusters |
| Negative | Samples likely in wrong cluster |

> A positive silhouette score on this dataset confirms that `alcohol` and `malic_acid` form meaningful natural groupings in the Wine data.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/kmeans-wine-clustering.git
cd kmeans-wine-clustering
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib notebook
```

### 3. Launch the notebook

```bash
jupyter notebook k_means_clustering_implementation.ipynb
```

---

## 💡 Key Concepts Covered

- Difference between **supervised** and **unsupervised** learning
- How K-Means iteratively assigns and updates cluster centroids
- Why **feature scaling** is mandatory for distance-based clustering
- Using the **Elbow Method** to select the optimal number of clusters
- Interpreting **Silhouette Score** as a cluster quality metric
- Visualizing high-dimensional data in 2D with selected features

---

## 🔗 Related Projects

| Project | Type | Algorithm | Dataset |
|---|---|---|---|
| [Naive Bayes](https://github.com/Mubinbilal/naive-bayes-breast-cancer) | Supervised | Gaussian NB | Breast Cancer |
| [SVM](https://github.com/Mubinbilal/svm-breast-cancer) | Supervised | Kernel SVM | Breast Cancer |
| [KNN](https://github.com/Mubinbilal/knn-digits-classifier) | Supervised | K-Nearest Neighbors | Digits |
| [Random Forest](https://github.com/Mubinbilal/random-forest-iris) | Supervised | Ensemble | Iris |
| [Decision Tree](https://github.com/Mubinbilal/decision-tree-iris) | Supervised | Tree-based | Iris |
| [Linear Regression](https://github.com/Mubinbilal/linear-regression-basic) | Supervised | Regression | Custom |

---

## 🙌 Acknowledgements

Built as part of a hands-on Machine Learning series. This is the first **unsupervised learning** project in the series, complementing the supervised classification and regression models built previously.
