# DRAR
This is our Pytorch implementation for the paper: DRAR: Diffusion-Based Relation Augmentation for Recommendation, (under review).

# Introduction
Graph neural network-based recommenders typically employ popular aggregation paradigms to learn representation from higher-order nodes within a graph. 
However, these simple aggregation paradigms have limitations in mitigating noise impacts and capturing complex user preferences. To address these limitations, some studies have attempted to enhance representation through contrastive augmentation across different views. Despite some effectiveness, the outcomes derived from simple view contrasts are suboptimal. Such approaches still face two significant challenges: 1) the influence of multivariate noise in interaction data and 2) knowledge biases introduced by irrelevant connections. 
In this study, we propose a novel Diffusion-Based Relation Augmentation for Recommendation (DRAR) to overcome the above challenges. First, we alleviate the impact of interaction noise by injecting uncertainty and generating preference distributions within our diffusion module. Next, we design a relation augmentation module to effectively capture user neighborhood-level and context-level enhanced representations to alleviate the knowledge bias of irrelevant connections. Furthermore, we design a collaborative alignment module that enhances the model's robustness by aligning user representation views at different stages. Extensive experiments on three public datasets consistently demonstrate the superiority of our model over the state-of-the-art approaches. For instance, on the MIND dataset, our model achieves significant improvements of 5.49% and 11.93% in Recall@20 and NDCG@20, respectively, compared to the strongest baseline.

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
