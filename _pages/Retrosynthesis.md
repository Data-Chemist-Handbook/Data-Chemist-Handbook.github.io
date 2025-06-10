---
title: 7. Retrosynthesis
author: Haomin
date: 2024-08-17
category: Jekyll
layout: post
---
dataset: USPTO subset (https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00064)

Organic synthesis, the design of synthetic routes to organic compounds, is a crucial discipline that holds relevance not only in the field of chemistry but across many other scientific domains and industries, including medicine and pharmaceuticals. It enables access to new molecules for a wide range of applications, including drug development and materials science.

The two core approaches involved in the synthesis of new molecules are: forward reaction prediction and retrosynthetic reaction prediction. Forward reaction prediction is the inference of the potential products of a given set of reactants, reagents, and reaction conditions. Retrosynthetic reaction prediction is simply the inverse process of that – starting with the target molecule and reasoning backward to determine how it might be constructed. It involves recursively deconstructing a target compound into simpler precursors that can ultimately be sourced or synthesized. This backward reasoning process that proposes plausible synthetic routes may yield multiple valid pathways for a single molecule. This one-to-many mapping reflects the complexity and flexibility of chemical synthesis. 

Retrosynthesis is a critical link between digital molecule design and real-world synthesis. Once the structure of a molecule is determined "in silo", retrosynthetic analysis helps determine whether and how it can be made using available chemical building blocks. In the context of drug discovery and materials science, this approach accelerates the development pipeline by guiding chemists toward feasible and efficient synthetic routes. The ability to identify valid reaction sequences with minimal experimentation is a valuable asset, especially when time or resources are limited.

## 7.1 Retrosynthetic Planning & Single-step Retrosynthesis
Complete retrosynthesis planning provides a series of reactions that sequentially breaks up a complex target molecule into smaller and simpler pieces until all of the pieces are commercially available.

This process is often represented as a tree structure: the root is the target molecule, and branches represent reaction steps that simplify the molecule progressively. The goal is to trace a path from the target to a set of readily available ingredients, ensuring that each intermediate step is chemically viable.

Each of these intermediate steps can be viewed as an independent retrosynthesis problem, referred to as single-step retrosynthesis. In other words, the retrosynthetic process can be broken down into discrete stages, each addressing a single chemical transformation. These are known as single-step retrosynthesis tasks. At each step, the target molecule is split into one or more simpler reactants. 

However, even inferring a single-step retrosynthesis is not trivial. Both it and retrosynthetic planning have historically relied on the expertise, domain-knowledge and experience of chemists, as well as costly trial and error.

## 7.2 Computational methods

## 7.3 Recurrent Neural Networks (RNNs) or LSTM or Transformer

## 7.4 Graph Neural Networks
