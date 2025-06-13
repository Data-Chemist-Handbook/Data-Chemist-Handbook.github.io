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

## 7.2 Computational Approaches to Retrosynthesis
The application of computational techniques to retrosynthesis analysis has emerged as an active and challenging area of research. In recent years, the accumulation of chemical synthesis data, coupled with advances in deep learning, has accelerated the development of computer-assisted synthesis processes, particularly for single-step retrosynthesis.

### 7.2.1 SMILES (Simplified Molecular Input Line Entry System)
Retrosynthesis prediction is complex and heavily dependent on molecular descriptors such as SMILES and molecular fingerprints. The Simplified Molecular Input Line Entry System (SMILES) is a string-based notation used to represent molecular structures and reactions. Each element in a SMILES string can be interpreted as a token in a machine translation model. By converting equivalent chemical structures into SMILES strings, several models can be employed for reaction prediction. Notable among these are the sequence-to-sequence (seq2seq) model and the Transformer model, based on attention mechanisms. These models will be discussed in detail in upcoming sections.

### 7.2.2 Types of Computational Approaches
Computational methods for retrosynthesis can be broadly categorized into two main types: template-based and template-free approaches.

#### Template-Based Methods
Template-based methods rely on predefined collections of reaction rules. hese methods typically treat single-step retrosynthesis as a classification task, selecting the most suitable reaction template to generate a given target product.

Despite their structured approach, template-based methods have notable limitations. They cannot predict retrosynthetic outcomes for target molecules that involve novel synthesis patterns not covered by the existing template library. Moreover, updating these libraries to incorporate newly discovered synthesis knowledge can be tedious and labor-intensive.

#### Template-Free Methods
In contrast, template-free methods predict the reactants of a target molecule without depending on predefined reaction templates. Because they are not constrained by a fixed rule set, these approaches are particularly valued for their ability to generalize to novel reactions and unfamiliar scenarios. Machine learning models leveraged in this approach include: 
- Seq2Seq models 
- Transformer-based models 
- Graph Neural Networks (GNNs)

### 7.2.3 Challenges in Template-Free Methods
While template-free methods facilitate the discovery of novel synthesis routes, they also introduce new challenges. One key issue is that the reactant SMILEs generated by these models may be chemically invalid or commercially unviable or unavailable. Although some studies have identified potential value in these invalid SMILES outputs, a substantial body of research has focused on mitigating this problem through mechanisms such as syntax post-checkers and semi-template-based methods.

Another limitation of SMILES-based methods is their inability to effectively capture molecules' structural information such as atomic properties, bond features, and adjacency relationships. In addition, current models struggle to fully exploit the potential of multiple molecular descriptors. This challenge often necessitates trade-offs between computational efficiency and predictive accuracy. For example, molecular fingerprinting emphasizes detailed structural features, while SMILES provides more global molecular information. When either descriptor is missing, important molecular characteristics may be lost.

Despite recent progress, the complexity of machine-supported retrosynthesis continues to motivate further research into data-driven strategies for synthesis planning.

A significant dimension yet to be fully addressed is empirical chemical knowledge, particularly the influence of catalysts and solvents as the same set of reactants can yield different products depending on the solvent or catalyst used. Considerations of reaction conditions, catalyst availability, and associated costs remain active and important areas for future research.

## 7.3 Seq2Seq LSTM 

The first machine learning-based template-free approach to retrosynthesis we will discuss is the use of a Seq2Seq LSTM. This is a sequence-to-sequence architecture built using two LSTMs, one serving as the encoder and the other as the decoder.
*(See Chapter 6 for information on RNN and LSTM, and Chapter 3 for information on neural networks.)*

The Seq2Seq LSTM architecture is used for tasks where one sequence of data needs to be transformed into another sequence, especially when the input and output sequences can have different lengths. Retrosynthesis can be framed as a sequence-to-sequence task, a "translation" from product to reactant, when molecules are represented using SMILES strings.

### 7.3.1 Choosing Seq2Seq LSTM for Retrosynthesis Tasks

While Seq2Seq LSTM is generally considered inferior to more modern approaches like Transformers or Graph Neural Networks (also presented in this chapter), it does have certain advantages:

- **Simple to implement and train:** Especially if working with tokenized SMILES strings (reactants >> product or vice versa).

- **Captures sequence dependencies:** LSTMs are good at modeling dependencies in sequences, which helps with SMILES syntax.

- **Works decently for small datasets:** On datasets like USPTO-50K, it gives reasonable results without huge compute.

- **Useful in resource-constrained settings:** It can be useful for prototyping, building lightweight models, or if constrained by compute and dataset size, or applying post-processing (e.g., syntax correction, beam search) to improve outputs.

Overall, Seq2Seq LSTM is a conceptually clean and easy-to-train baseline for retrosynthesis prediction using SMILES strings.

### 7.3.2 Seq2Seq LSTM for Retrosynthesis: Application Code

## 7.4 Transformer

## 7.5 Graph Neural Networks
