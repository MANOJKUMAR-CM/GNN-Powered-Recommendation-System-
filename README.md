# 🛒 H&M Graph Neural Network Recommendation System

This project builds a **Graph Neural Network (GNN)-based recommendation system** on the **H&M Personalized Fashion Recommendations dataset (Kaggle)**.  

Customers and articles are modeled as a **heterogeneous bipartite graph**, and a GNN (GraphSAGE) is trained to perform **link prediction** for personalized product recommendations.

---

## 🚀 Project Overview

- **Dataset**: [H&M Personalized Fashion Recommendations (Kaggle)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)  

- **Graph Construction**:  
  - **Nodes**:  
    - `customer`: 1,371,980 nodes with 9-dimensional features (demographics & embeddings)  
    - `article`: 105,542 nodes with 683-dimensional features (metadata & embeddings)  
  - **Edges**: 31,788,324 purchase interactions (customer ↔ article), with time and edge attributes  

- **Downsampled Graph** (for demonstration / compute efficiency):  
  - Customers: 100,000  
  - Articles: 82,477  
  - Edges: 2,327,067 interactions  

- **Model**:  
  - Implemented **Heterogeneous Graph Neural Networks** with PyTorch Geometric  
  - Experimented with **GraphSAGE** layers  
  - Learned refined embeddings for customers & articles  

- **Task**: **Link prediction** → Predict if customer *X* will purchase article *Y*  

---

## 📊 Results

| Metric              | Score  |
|----------------------|--------|
| **Accuracy**         | 0.8475 |
| **ROC-AUC**          | 0.9170 |
| **Average Precision**| 0.9101 |
| **F1-score**         | 0.89   |

- Strong performance on a **large-scale real-world dataset** (>2M interactions).
- Outperforms simple baselines like LightGBM/ matrix factorization models.  

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Libraries**:  
  - [PyTorch](https://pytorch.org/)  
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)  
  - [Scikit-learn](https://scikit-learn.org/)  

---

## ✨ Key Highlights

- Handles **heterogeneous graph structure** (customer ↔ article bipartite graph).  
- Incorporates **temporal purchase information** and **article/customer metadata**.  
- Scalable to **millions of nodes and edges**.  
- Demonstrates application of **Graph Neural Networks in personalized recommendations**.  

---

## 🔮 Future Work

- Add **Hit@K / Recall@K** evaluation for top-K recommendation quality  
- Explore **temporal graph models** (e.g., TGAT, TGN)  
- Deploy trained model as an **API-based recommender service**  

---
## 📧 Contact

If you have questions, suggestions, or just want to connect, feel free to reach out!

- **Name**: Manoj Kumar.CM  
- **Email**: [manoj.kumar@dsai.iitm.ac.in]  
- **GitHub Profile**: [Manoj Kumar C M](https://github.com/MANOJKUMAR-CM)
---  
💡 **Have feedback or suggestions? Feel free to contribute!**  


