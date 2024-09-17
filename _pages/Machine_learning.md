---
title: 3. Machine Learning Models
author: Haomin
date: 2024-08-13
category: Jekyll
layout: post
---

Machine learning (ML) is a subfield of artificial intelligence (AI) focused on developing algorithms and statistical models that enable computers to learn from and make decisions based on data. Unlike traditional programming, where explicit instructions are given, machine learning systems identify patterns and insights from large datasets, improving their performance over time through experience.

ML encompasses various techniques, including supervised learning, where models are trained on labeled data to predict outcomes; unsupervised learning, which involves discovering hidden patterns or groupings within unlabeled data; and reinforcement learning, where models learn optimal actions through trial and error in dynamic environments. These methods are applied across diverse domains, from natural language processing and computer vision to recommendation systems and autonomous vehicles, revolutionizing how technology interacts with the world.

## 3.1 Decision Tree and Random Forest

Decision Trees and Random Forests are powerful machine learning techniques used for classification and regression tasks.

Decision Trees are intuitive models that split data into branches based on feature values, forming a tree-like structure. Each internal node represents a decision based on a feature, each branch represents the outcome of that decision, and each leaf node represents a class label or regression value. The goal is to create a model that predicts the target variable by learning simple decision rules inferred from the features. Decision trees are easy to interpret and visualize, but they can be prone to overfitting, especially with complex datasets.

Random Forests address this limitation by combining multiple decision trees into an ensemble. A Random Forest builds a collection of decision trees, each trained on a different subset of the data and with random subsets of features. This process, known as bagging (Bootstrap Aggregating), helps to reduce variance and improve the model's robustness. The final prediction is made by aggregating the predictions of all individual trees, typically through majority voting for classification tasks or averaging for regression tasks. Random Forests generally perform better than individual decision trees by balancing model complexity and generalization.

## 3.2 Neural Network

A neural network is a computational model inspired by the neural structure of the human brain, designed to recognize patterns and learn from data. It consists of layers of interconnected nodes, or neurons, which process input data through weighted connections.

Structure: Neural networks typically include an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to neurons in the adjacent layers. The input layer receives data, the hidden layers transform this data through various operations, and the output layer produces the final prediction or classification.

Functioning: Data is fed into the network, where each neuron applies an activation function to its weighted sum of inputs. These activation functions introduce non-linearity, allowing the network to learn complex patterns. The output of the neurons is then passed to the next layer until the final prediction is made.

Learning Process: Neural networks learn through a process called training. During training, the network adjusts the weights of connections based on the error between its predictions and the actual values. This is achieved using algorithms like backpropagation and optimization techniques such as gradient descent, which iteratively updates the weights to minimize the prediction error.

## 3.3 Graph Neural Network

Graph Neural Networks (GNNs) are a class of neural networks designed to operate on graph-structured data. Unlike traditional neural networks, which work with data in grid-like structures (such as images or sequences), GNNs are specifically tailored to handle data represented as graphs, where entities are nodes and relationships are edges.

Graph Structure: A graph consists of nodes (vertices) and edges (connections between nodes). GNNs are adept at processing and learning from this structure, capturing the dependencies and interactions between nodes.

Message Passing: GNNs typically operate through a message-passing mechanism, where nodes aggregate information from their neighbors to update their own representations. This involves sending and receiving messages along the edges of the graph and combining these messages to refine the node's feature representation.

Layer-wise Propagation: In a GNN, the learning process involves multiple layers of message passing. Each layer updates node features based on the aggregated information from neighboring nodes. This iterative process allows the network to capture higher-order relationships and global graph patterns.

Advantages: GNNs leverage the inherent structure of graph data, making them powerful for tasks involving complex relationships and dependencies. They can model interactions between entities more naturally than traditional neural networks and are capable of handling graphs of varying sizes and structures.
