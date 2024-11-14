# DRAR
This is our Pytorch implementation for the paper: DRAR: Diffusion-Based Relation Augmentation for Recommendation, (under review).

# Introduction
Graph neural network-based recommenders employ the aggregation paradigms to learn node representation from higher-order neighboring nodes within the graph. However, these simple aggregation paradigms may perform poorly when mitigating noise impacts and capturing complex user preferences. To address it, some studies have attempted to enhance representation through contrastive augmentation across different views. Despite some effectiveness, the simple-view contrasts are still suboptimal with some unresolved significant challenges: 1) the influence of multivariate noise in interaction data and 2) knowledge biases introduced by irrelevant connections, and 3) user's multiple interests. 
In this work, we propose a novel method named \textbf{D}iffusion-Based \textbf{R}elation \textbf{A}ugmentation for knowledge-aware \textbf{R}ecommendation (short as DRAR) to overcome the above challenges. First, we alleviate the impact of interaction noise by injecting uncertainty and generating preference distributions with a diffusion-based module. Next, we design a relation augmentation module to effectively capture user neighborhood-level and context-level enhanced representations to alleviate the knowledge bias of irrelevant connections. Furthermore, we design a collaborative alignment module that enhances the model's robustness by aligning user representation views at different stages. Extensive experiments on three benchmark datasets consistently demonstrate the superiority of our model over the state-of-the-art approaches. Our model demonstrate average improvements of 6.78\% in Recall and 7.38\% in NDCG across all datasets.

# Requirement
pytorch==1.10.1
numpy==1.21.6
scikit-learn==1.0.2

# Usage
The hyper-parameter search range and optimal settings have been clearly stated in the codes.

Train and Test
python main.py 

# Dataset
We provide three processed datasets: Book-Crossing, MIND, and Last.FM.
