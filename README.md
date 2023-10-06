# Federated Multi-Task Learning
---
Federated learning is new ML technique that uses multiple independent sessions (nodes), each with its own dataset, to train a model.

FL poses new statistical and systems challenges, namely:
- Since each node has its own dataset, data on each node may follow different distributions. Also, the number of data points may vary significantly, and there may be an underlying structure that captures the relationship between nodes and their distributions.
- As the number of nodes in the network increases, communication speed becomes a bottleneck.
- The storage, computational, and communication capacities of each node may differ, which makes the network less fault tolerable.

## Importance of This Algorithm
The authors mention a number of potential applications of federated learning: learning sentiment, semantic location, activities of mobile phone users, predicting health events like low blood sugar or heart attach risk, or detecting burglaries.

In general, this algorithm is important because it is an advanced version of classical distributed learning.

## Main Idea of the Approach
Recently, storage and computational power of modern devices have increased signficantly, so it is more appealing to push data storage and computations to the network edge.

Authors propose to learn separate models for each node through multi-task learning (MTL), as opposed to training a single global model across the network.

## What It Is Based on
This paper is based on three ideas:
1. Learning Beyond the Data Center - training ML models locally on distributed networks rather than centrally. This approach is now available as the computational power is growing. However, existing solutions cannot deal tith non-IID data and are not fault-tolerable
2. Multi-Task Learning. It involves learning models for multiple related tasks simultaneously. Current solutions can work with non-IID and imbalanced data, but are not suited for distributed systems.
3. Distributed Multi-Task Learning - a new area of research that still cannot balance well between computation and communication; as a result this approach has low fault tolerance. However, a method that leverages the distirbuted framework CoCoA was proposed.

CoCoA is a special case of a more general approach proposed by the authors - Mocha.

## Key Features of the Proposed Method
The only key feature of the proposed Mocha method is that it can handle the unique systems challenges of the federated environment.

Instead of waiting for responses from all workers before performing a synchronous update (which leads too problems), it allows nodes to approximately solve their subproblems, where the quality of approximation is controlled by a per-node parameters.

Thus, it creates a framework for Federated Multi-Task Learning.

## Intuition Behind the Method


## Improvements over the Basic Algorithms
