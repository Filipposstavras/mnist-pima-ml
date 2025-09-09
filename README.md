# MNIST & Pima Indians — Machine Learning Project

Applied machine learning project with two problems:
- **MNIST Classification & Clustering**  
- **Pima Indians Diabetes Classification**  

Demonstrates supervised learning, ensemble methods, dimensionality reduction, clustering, and semi-supervised learning.

---

## Problem 1 — MNIST Classification & Clustering

**Goal:** Explore supervised and unsupervised approaches on MNIST digits.  
**Dataset:** [MNIST (OpenML)](https://www.openml.org/d/554)

**Methods:**
- Decision Tree Classifier with Grid Search (max_features, max_depth)  
- PCA (90% variance) + Decision Tree  
- Gradient Boosting Classifier (shallow trees)  
- PCA Reconstruction of digits  
- KMeans clustering (20 clusters) with centroid visualization  
- Cluster-based labeling via nearest centroid → accuracy & F1  

**Key results:**  
- Best DT: accuracy ≈ 66.7%, F1 ≈ 65.7%  
- PCA+DT lower accuracy (≈56%)  
- Gradient Boosting outperformed with ≈78.6% accuracy  
- Clustering-based labeling: ≈4.5% accuracy  

---

## Problem 2 — Pima Indians Diabetes Classification

**Goal:** Detect diabetes from clinical features.  
**Dataset:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Steps:**
- Handle missing values (replace zeroes with median for relevant columns)  
- Stratified split (700 train / rest test)  
- Classifiers:
  - Decision Tree  
  - Random Forest  
  - Bagging (SVM base)  
  - AdaBoost (Decision Tree base)  
- Compare precision, recall, F1, confusion matrices  
- Semi-supervised learning:
  - Random Forest as base
  - SelfTrainingClassifier with threshold=0.99
  - Compare against supervised model trained on 200 labeled samples  

**Key results:**  
- Random Forest best: accuracy ≈ 82%  
- Semi-supervised RF ≈ 70.6% vs supervised ≈ 73.5% (200 labels)  
- Semi-supervised underperformed supervised  

---

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
