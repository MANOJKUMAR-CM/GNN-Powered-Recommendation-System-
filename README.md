# 🛒 H&M Graph Neural Network Recommendation System

This project builds a **Graph Neural Network (GNN)-based recommendation system** on the **H&M Personalized Fashion Recommendations dataset**.  
Customers and articles are modeled as a **heterogeneous bipartite graph**, and a GNN (GraphSAGE / GCN) is trained to perform **link prediction** for personalized product recommendations.

---

## 🚀 Project Overview
- **Dataset**: [H&M Personalized Fashion Recommendations (Kaggle)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)  
- **Graph Construction**:
  - **Nodes**:  
    - Customers (features: demographics, embeddings → 9-dim)  
    - Articles (features: metadata, embeddings → 683-dim)  
  - **Edges**: Customer → Article purchases/interactions  
- **Model**:  
  - Implemented **Heterogeneous Graph Neural Networks** with PyTorch Geometric  
  - Experimented with **GraphSAGE** and **GCN** layers  
  - Learned refined embeddings for customers & articles  
- **Task**: **Link prediction** (will customer *X* buy article *Y*?).  

---

## 📊 Results

| Metric              | Score  |
|----------------------|--------|
| **Accuracy**         | 0.8475 |
| **ROC-AUC**          | 0.9170 |
| **Average Precision**| 0.9101 |
| **F1-score**         | 0.85   |

✅ Strong performance on a large-scale dataset (>2M interactions).  
✅ Outperforms simple baselines like matrix factorization or collaborative filtering.  

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**:  
  - [PyTorch](https://pytorch.org/)  
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)  
  - Scikit-learn (metrics)  
  - Pandas, NumPy  

---

## 📂 Repository Structure
