---
title: 4. Molecular Property Prediction
author: Alex, Haomin
date: 2024-08-14
category: Jekyll
layout: post
---

In modern chemistry, predicting the properties of a molecule before synthesizing it is both a challenge and an opportunity. Experimental testing can be time-consuming, expensive, or even infeasible for large compound libraries. As a result, computational methods have become essential for prioritizing candidates with desired characteristics.

Molecular property prediction focuses on learning the relationship between a molecule’s structure and its physicochemical or biological properties. These properties can range from solubility, toxicity, and permeability to more complex metrics like bioavailability or metabolic stability. The goal is to build models that can accurately predict these properties based on the molecular information available—often in the form of descriptors, fingerprints, or SMILES strings.

Machine learning models are particularly well-suited for this task because they can capture non-linear, high-dimensional relationships between structure and function. Chemists can now leverage a wide array of algorithms to train predictive models using labeled datasets of molecules and their known properties.

This chapter explores how various machine learning techniques—including Recurrent Neural Networks, Graph Neural Networks, Random Forests, and traditional Neural Networks—can be applied to property prediction problems in chemistry. These methods are tested on benchmark datasets such as **BBBP**, which evaluates a compound’s ability to cross the blood–brain barrier.

Whether you’re screening drug candidates, designing safer chemicals, or optimizing synthetic routes, molecular property prediction provides a critical edge. When used effectively, it enables smarter experimentation, reduces waste, and accelerates discovery.

We begin with sequence-based learning using **Recurrent Neural Networks** for SMILES classification.

## 4.1 Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a type of deep learning architecture designed to process sequential data—making them especially powerful for tasks involving temporal order or structural dependencies. In chemistry, this makes RNNs a natural fit for working with **SMILES strings**, a linear representation of molecules that encodes structural information using a sequence of characters.

Unlike traditional feedforward neural networks, which treat each input independently, RNNs maintain a **hidden state** that carries contextual information from one step of the sequence to the next. This allows them to capture dependencies across a molecular structure that are expressed sequentially, such as ring closures or branches that appear far apart in the SMILES string but are chemically related.

---

#### Why Use RNNs for SMILES?

SMILES strings are inherently sequential—they describe atoms and bonds in a specific order. For example:

```
CC(=O)O
```

This string encodes acetic acid: a methyl group (C), a carbonyl (=O), and a hydroxyl (O). To understand or predict molecular properties from this input, a model must consider not just which characters are present, but **how they relate to one another over the course of the sequence**.

RNNs are designed to do exactly that. They process one character at a time while maintaining a memory of previous characters, enabling the model to learn how patterns in molecular syntax relate to the molecule’s function or properties.

---

#### RNN Architecture Basics

An RNN processes a sequence step by step. At each time step *t*, it:

- Receives an input token (e.g. a SMILES character)
- Updates its hidden state based on the current input and the previous hidden state
- Optionally generates an output (e.g. a prediction or classification)

Mathematically:

$$
h_t = f(W \cdot x_t + U \cdot h_{t-1} + b)
$$

Where:

- \( h_t \) is the hidden state at time *t*
- \( x_t \) is the current input
- \( W, U \) are learned weight matrices
- \( b \) is a bias term
- \( f \) is typically a nonlinear activation function (like tanh or ReLU)

Because the model shares parameters across time steps, RNNs can generalize to variable-length sequences—an important property for molecules, which naturally vary in size.

---

#### Practical Variants: LSTM and GRU

Standard RNNs suffer from a problem called **vanishing gradients**, which makes it hard to learn long-range dependencies. To address this, more advanced versions like **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** were developed.

These architectures include internal gating mechanisms that control how much past information to retain or forget. In molecular modeling, they can capture subtle long-distance relationships, such as how atoms at opposite ends of a molecule may influence solubility or reactivity.

---

#### What Can RNNs Learn About Molecules?

- **Property Prediction**: RNNs can be trained to classify molecules based on solubility, toxicity, bioavailability, etc., using their SMILES strings as input.
- **Molecule Generation**: Trained RNNs can generate new valid SMILES strings, enabling data-driven de novo molecule design.
- **Reaction Prediction**: When given reactants as input, RNNs can learn to output the likely product SMILES.

These applications rely on the model's ability to learn chemical syntax as a language—very similar to how RNNs are used in natural language processing.

---

### 4.1.1 BBBP Dataset Classification Using RNNs

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/1p04CBncUdBVaSiQDW05AjKXa95mPN5q1?usp=sharing)

To demonstrate how Recurrent Neural Networks (RNNs) can be used for molecular property prediction, we'll classify molecules from the BBBP dataset based on whether they are likely to cross the blood–brain barrier (BBB). This is a critical consideration in drug design, as compounds that cannot reach the central nervous system are ineffective for neurological targets.

---

#### Problem Statement
Given a molecule’s SMILES string, our task is to predict a binary label:

- `1` if the molecule is BBB-permeable

- `0` otherwise

This is a **sequence classification problem**. We'll use a **Gated Recurrent Unit (GRU)** model that reads SMILES character-by-character and learns chemical structure patterns over time.

---

#### Dataset Overview
We use the curated BBBP dataset hosted on GitHub:

[https://raw.githubusercontent.com/Data-Chemist-Handbook/Data-Chemist-Handbook.github.io/refs/heads/master/_pages/BBBP.csv](url)

The dataset contains:

- A `smiles` column: molecular structure in SMILES format
- A `p_np`   column: binary permeability label (`1` for permeable, `0` for not)

---

#### Example: Classifying Molecules with GRU

This code is designed to run in Google Colab and performs the full classification pipeline:

```python
# Step 1: Install dependencies
!pip install -q rdkit pandas scikit-learn tensorflow

# Step 2: Load the BBBP dataset from GitHub
import pandas as pd
url = "https://raw.githubusercontent.com/Data-Chemist-Handbook/Data-Chemist-Handbook.github.io/refs/heads/master/_pages/BBBP.csv"
data = pd.read_csv(url)

# Step 3: Extract SMILES and labels
smiles_list = data['smiles']
labels = data['p_np']

# Step 4: Tokenize SMILES at character level
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles_list)
sequences = tokenizer.texts_to_sequences(smiles_list)
X = pad_sequences(sequences, padding='post', maxlen=120)
y = labels.values

# Step 5: Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build GRU model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=120))
model.add(GRU(units=64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Step 8: Evaluate performance
loss, acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {acc:.2f}')
```

---

#### Analysis

This example demonstrates how RNNs can directly process SMILES sequences to learn patterns relevant to pharmacological properties—specifically, blood–brain barrier (BBB) permeability. During training:

- The **embedding layer** transforms each character in the SMILES string into a dense vector representation, allowing the network to learn chemical syntax.
- The **GRU (Gated Recurrent Unit)** processes the sequence of vectors, retaining context across the string and modeling relationships between distant atoms.
- The **final dense layer** outputs a probability estimate indicating whether the molecule is likely to cross the blood–brain barrier.

In this basic configuration, the model achieved around **79% test accuracy** after 5 epochs. While this is a strong baseline, further improvements can be made by:

- Adding **dropout layers** to reduce overfitting
- Increasing the number of **GRU units** or stacking multiple layers
- Using **learning rate schedules** or more advanced optimizers

This reinforces the utility of RNNs for tasks involving molecular structure encoded in sequence form, and highlights the accessibility of SMILES-based deep learning workflows in cheminformatics.

---

#### Key Takeaways
- RNNs are capable of learning meaningful features directly from raw SMILES text.
- Character-level tokenization ensures syntactic fidelity of molecular input.
- GRU models are efficient for learning dependencies in molecular sequences.
- This approach enables rapid screening of drug candidates for CNS activity based on SMILES alone.

---

### 4.1.2 Understanding How RNNs Interpret Molecular Sequences

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/18qXcfimZM0bm1DH1gzKSX9OWTq2TFOyn?usp=sharing)

Recurrent Neural Networks (RNNs), especially their GRU and LSTM variants, are often treated as black boxes. But to truly harness their potential for molecular tasks, it's important to understand how they interpret the input SMILES sequences and build internal representations of molecular features.

---

#### From Characters to Chemistry: What RNNs Actually Learn
Each SMILES string is treated as a sequence of characters—such as `C`, `=`, `O`, `(`, or digits used to indicate ring closures. Initially, these characters are meaningless to the model. However, during training:
- The **embedding layer** transforms each character into a dense vector that captures how frequently it appears with other characters.
- As training progresses, characters like `'C'`, `'O'`, `'N'` begin to take on **chemical context**, because the model sees that certain patterns—such as `'C(=O)O'` (carboxylic acid) or `'N(C)C'` (tertiary amine)—correlate with particular output labels.

Thus, the network gradually learns to interpret chemically relevant **motifs** as sequential patterns.

---

#### Hidden State = Molecular Memory
At every step in a SMILES string, an RNN computes a hidden state vector that summarizes everything it has "seen" so far. This hidden state evolves as the sequence is read character by character.

In our BBBP example:

- Early in the sequence, the model might only have seen `'C'` or `'CC('`.
- As it processes more atoms and branches, it builds a richer internal representation that reflects possible **substructures, functional groups, or steric motifs**.
- When the model reaches the end of the SMILES, it uses its final hidden state to decide if the molecule is likely to be BBB-permeable.

This makes RNNs especially powerful for molecules with **long-range dependencies**, such as ring closures or functional groups connected through non-adjacent atoms.

---

#### Attention Mechanisms

In more advanced architectures, **attention layers** can be added to help the model decide which parts of the SMILES string are most important for making a prediction. While not used in our basic GRU model, attention is widely used in modern molecular sequence models (including Transformers).

---

#### Visualizing Learned Representations (Optional Exploration)
You can visualize what the model is learning using:

- **Embedding heatmaps:** Show which characters are embedded similarly.
- **Hidden state trajectories:** Plot hidden state values as the SMILES is processed to detect “activation spikes” around chemical groups.
- **Saliency maps (advanced):** Highlight which characters in the input had the greatest effect on the model's prediction.

---

#### Practice Problem : Exploring RNN Hidden States
**Instructions**
Use the trained GRU model from 4.1.1 and modify it to return the hidden states of the SMILES sequence `CC(=O)O` (acetic acid). Specifically, extract the hidden state after each character is read to visualize how the model "builds" its chemical understanding step-by-step.

**Expected Learning Outcome**
This exercise will help you understand how RNNs accumulate sequential information and demonstrate that even short molecules result in meaningful internal representations.This exercise will help you understand how RNNs accumulate sequential information and demonstrate that even short molecules result in meaningful internal representations.

**Solution Code:**
```python
# Install dependencies (if needed)
!pip install -q rdkit pandas scikit-learn tensorflow

# Load the BBBP dataset
import pandas as pd
url = "https://raw.githubusercontent.com/Data-Chemist-Handbook/Data-Chemist-Handbook.github.io/refs/heads/master/_pages/BBBP.csv"
data = pd.read_csv(url)

# Tokenize SMILES
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

smiles_list = data['smiles']
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles_list)

# Create vocab_size for embedding
vocab_size = len(tokenizer.word_index) + 1
```

Once that’s run, your tokenizer and vocab_size variables will be defined, and then you can safely run your visualization code:

```python
# Your visualization code
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Prepare test input
test_smiles = ['CC(=O)O']
test_seq = tokenizer.texts_to_sequences(test_smiles)
test_pad = pad_sequences(test_seq, padding='post', maxlen=120)

# Define GRU model to return all hidden states
input_layer = Input(shape=(120,))
embed_layer = Embedding(input_dim=vocab_size, output_dim=32)(input_layer)
gru_layer, state_seq = GRU(units=64, return_sequences=True, return_state=True)(embed_layer)

# Build model and predict
intermediate_model = Model(inputs=input_layer, outputs=gru_layer)
hidden_states = intermediate_model.predict(test_pad)

# Plot hidden state activations
plt.figure(figsize=(14, 6))
plt.imshow(hidden_states[0][:len(test_seq[0])], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("GRU Hidden State Activations for 'CC(=O)O'")
plt.xlabel("Hidden State Dimension")
plt.ylabel("Sequence Position (Character Index)")
plt.xticks(np.arange(0, 64, 8))
plt.yticks(np.arange(len(test_seq[0])), labels=list(test_smiles[0]))
plt.show()
```

---

#### Expected Output
You will see a heatmap where:

- The **x-axis** shows the 64 GRU hidden state dimensions
- The **y-axis****** shows each character in the input SMILES string
- Color intensity shows how active each hidden unit is at each character step

---

#### Interpretation
This visualization offers an intuitive look at what your RNN is doing internally. For example:
- Repeating characters like `'C'` may produce similar activations early on.
- When the model encounters a `'('` or `'='`, you may notice distinct changes in hidden state activity, reflecting structural divergence.
- The character `'O'` at the end (representing a hydroxyl) may trigger stronger activations in certain units—this could correlate with features relevant to solubility or permeability.

---

#### Takeaway
The GRU model doesn’t just memorize chemical strings—it **learns to encode chemistry** in a way that preserves the molecule’s structural logic over time. This exercise demonstrates how sequence-based neural networks build internal representations step-by-step, capturing both local and global features in a molecule.

---

### 4.1.3 Encoding and Embedding Strategies for SMILES

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/19NRosSmfuVg3s9hwDF-lFsgviHe_WcUR?usp=sharing)

When working with SMILES strings in deep learning models, how you represent each character or token can significantly affect model performance. This process—called encoding—translates the textual SMILES sequence into numerical inputs that a neural network can process.

---

#### One-Hot Encoding: Binary Vectors for Each Character

There are two primary strategies:

- **One-hot encoding**
- **Learnable embeddings (dense vector representations)**

In one-hot encoding, each character is represented by a binary vector. For a vocabulary of size *V*, a single character is mapped to a vector of length *V* where only one element is 1 (the index of that character), and the rest are 0.

For example, if your SMILES vocabulary contains `[C, O, N, (, ), =]`, then:

- `'C'` → `[1, 0, 0, 0, 0, 0]`
- `'='` → `[0, 0, 0, 0, 0, 1]`

This method is:

- Simple and easy to implement
- Sparse, meaning most of the vector values are zeros
- Memory-intensive, especially for large vocabularies

Most importantly, one-hot vectors treat all characters as equally distinct. `'C'` and `'N'` have no more similarity than `'C'` and `'='`, even though chemically they might share more context.

---

#### Learnable Embeddings: Letting the Model Discover Chemistry

In modern neural networks, we often use an embedding layer instead of one-hot vectors. This layer learns to map each character to a dense, continuous vector of fixed length (e.g., 32 or 64 dimensions) during training.

**Benefits:**

- **Efficient:** Lower-dimensional vectors are less memory-intensive
- **Expressive:** Embeddings can capture chemical relationships between symbols (e.g., `'O'` and `'N'` might end up closer in vector space than `'O'` and `'('`)
- **Trainable:** The network learns to organize the embedding space based on the prediction task (e.g., solubility, toxicity)

In Keras, this is done with:

```python
Embedding(input_dim=vocab_size, output_dim=32)
```

Here, `input_dim` is the total number of SMILES characters, and `output_dim` is the embedding size.

---

#### One-Hot vs. Embedding: Which Should Chemists Use?

| Strategy  | Pros                          | Cons                               |
| --------- | ----------------------------- | ---------------------------------- |
| One-hot   | Interpretable, no parameters  | Sparse, doesn't capture similarity |
| Embedding | Compact, learns relationships | Requires more data to train well   |

For most deep learning tasks, **learnable embeddings are preferred**. They allow models to generalize more effectively across chemical space and are standard in cheminformatics applications today.

---

#### Practice Problem: Compare One-Hot vs. Embedding

**Instructions**
Compare model performance using one-hot encoding vs. an embedding layer on a subset of the BBBP dataset:

1. Modify your GRU model from Section 4.1.1 to use one-hot encoded SMILES instead of an embedding layer.
2. Train the model for 5 epochs and note the test accuracy.
3. Repeat using the embedding layer (as done originally).
4. Which method gives better performance? Why?

**Solution Code:**

```python
# Step 1: Install and import dependencies
!pip install -q rdkit pandas scikit-learn tensorflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Step 2: Load the BBBP dataset
url = "https://raw.githubusercontent.com/Data-Chemist-Handbook/Data-Chemist-Handbook.github.io/refs/heads/master/_pages/BBBP.csv"
data = pd.read_csv(url)

smiles_list = data['smiles']
labels = data['p_np']
y = labels.values

# Step 3: Tokenize SMILES and pad
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles_list)
sequences = tokenizer.texts_to_sequences(smiles_list)
X_seq = pad_sequences(sequences, padding='post', maxlen=120)
vocab_size = len(tokenizer.word_index) + 1

# Step 4: One-hot encode the sequences
X_onehot = to_categorical(X_seq, num_classes=vocab_size)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.2, random_state=42)

# Step 6: GRU model without embedding
model = Sequential()
model.add(GRU(units=64, input_shape=(120, vocab_size)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 7: Train and evaluate
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy (One-Hot): {acc:.2f}")
```

---

#### Expected Output

You should see something like:

- One-hot model accuracy: \~76–78%
- Embedding model accuracy: \~79–81%

The embedding model usually performs better due to its ability to learn nuanced relationships between characters and their structural roles.

---

#### Takeaway

- One-hot encoding is simple but limited in chemical expressiveness.
- Learnable embeddings improve generalization, compactness, and structural understanding.
- In molecular property prediction, embeddings are the standard approach for RNN-based models.

---

### 4.1.4 Challenges of RNNs in Chemistry

Recurrent Neural Networks (RNNs) — especially GRUs and LSTMs — have demonstrated strong performance in SMILES-based molecular property prediction. However, their application in cheminformatics is not without limitations. In this section, we highlight key challenges chemists should consider when using RNNs on molecular data.

---

#### 1. Sensitivity to SMILES Canonicalization

RNNs treat SMILES strings as character sequences. However, the same molecule can be written in multiple valid SMILES formats:

```text
CC(=O)O          # Acetic acid
OC(=O)C          # Acetic acid (non-canonical)
```

**Problem:** The model may fail to generalize across equivalent molecules.

**Solution:** Always canonicalize SMILES using RDKit or similar tools before training to ensure consistency.

---

#### 2. Lack of Structural Awareness

SMILES encodes 2D molecular structure into 1D text. While RNNs can learn syntactic patterns, they do not inherently understand:

- Atom connectivity
- Ring structures
- Branching
- Stereochemistry (unless explicitly encoded)

**Chemist's Insight:** An RNN doesn't "see" a molecule — it reads it like a sentence. Important 3D or topological features may be lost unless explicitly encoded.

---

#### 3. Difficulty Capturing Long-Range Dependencies

In SMILES, chemically related atoms may be far apart in the sequence (e.g., ring closures or branching).

While GRUs and LSTMs help retain memory over longer sequences, they still struggle with non-local dependencies — especially in large molecules or those with fused rings.

---

#### 4. Computational Cost of Sequence Models

RNNs process one token at a time, making them:

- Slower to train than CNNs or Transformers
- More prone to vanishing gradients

For large-scale datasets or high-throughput screening, this can be a bottleneck.

---

#### 5. Limited Interpretability

RNNs lack built-in mechanisms for feature importance, making their predictions harder to interpret.

* Tree-based models offer clearer rationale
* Interpretability tools (e.g., attention, saliency) require additional components

---

#### Summary Table: Pros vs. Cons of RNNs for SMILES

| Aspect                  | Strengths                            | Limitations                                     |
| ----------------------- | ------------------------------------ | ----------------------------------------------- |
| Sequence Modeling       | Naturally fits SMILES format         | May miss spatial/graph context                  |
| Memory                  | GRUs/LSTMs capture history           | Struggles with long-range chemical dependencies |
| Representation Learning | Learns chemical syntax automatically | Input must be carefully canonicalized           |
| Efficiency              | Lightweight compared to Transformers | Slower than parallel models                     |
| Interpretability        | Embeds complex relationships         | Requires extra design for explanation           |

---

#### When Should Chemists Use RNNs?

| Recommended When...                     | Avoid When...                              |
| ----------------------------------------- | -------------------------------------------- |
| You're working with SMILES-only datasets  | You need graph- or 3D-aware modeling         |
| You want a simple, fast baseline model    | You require interpretability or explanations |
| Your task is classification or regression | Your molecules vary widely in size/branching |
| You have limited computational resources  | You're working with very large datasets      |

---

### 4.1.5 Embedding Alternatives – Comparing RNNs to Other Sequence Models

While Recurrent Neural Networks (RNNs) offer a natural way to model SMILES strings, they are no longer the only — or even the best — option for all molecular sequence tasks. In recent years, alternative neural architectures have emerged that outperform RNNs in both speed and accuracy, especially on larger datasets or tasks requiring global context.

This section introduces and compares several popular alternatives to RNNs for learning from molecular sequences.

---

#### 1. 1D Convolutional Neural Networks (CNNs)

Originally developed for image processing, 1D CNNs can also handle sequential data by applying filters over fixed-size windows of characters.

**How it works for SMILES:**

- The SMILES sequence is treated as a string of character embeddings.
- Convolutional filters learn substructure patterns (e.g., functional groups) across windows of characters.
- Multiple filters can detect different chemical motifs.

**Advantages:**

- Highly parallelizable — much faster to train than RNNs.
- Effective at learning local patterns in chemical syntax.
- Less prone to vanishing gradients.

**Limitations:**

- Lacks memory of order beyond the convolution window.
- Does not naturally model long-range dependencies or molecule-wide context.

**Chemist's Analogy:** CNNs are like a chemist scanning molecules for familiar substructures — they excel at detecting functional groups, but may miss the big picture.

---

#### 2. Transformers

The Transformer architecture, popularized by models like BERT and GPT, uses self-attention to relate every token to every other token in a sequence — regardless of distance.

**Why Transformers are useful for SMILES:**

- Attention layers can learn interactions between distant atoms in a molecule.
- They are highly parallelizable and scale well with large datasets.
- Pretrained models (like ChemBERTa or SMILES-BERT) can be fine-tuned for new tasks.

**Advantages:**

- State-of-the-art performance on many molecular property prediction tasks.
- Capture global context more effectively than RNNs or CNNs.
- Enable pretraining + fine-tuning workflows for transfer learning.

**Limitations:**

- Require more memory and compute than RNNs or CNNs.
- May need large datasets or pretraining to perform well.

**Chemist's Analogy:** Transformers are like a chemist reading the entire molecule at once, comparing every part with every other — a full-context thinker.

---

#### 3. Comparison Summary

| Architecture     | Key Strength                                     | Weakness                                 | When to Use                                            |
| ---------------- | ------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------ |
| RNN / GRU / LSTM | Good for small/medium SMILES datasets            | Slow training, struggles with long-range | When sequence order matters or datasets are small      |
| 1D CNN           | Fast training, effective local pattern detection | Misses global context                    | When looking for substructure-based patterns           |
| Transformer      | Best global modeling, state-of-the-art accuracy  | Computationally expensive                | For large-scale tasks, or when using pretrained models |

---

#### 4. Choosing the Right Sequence Model in Practice

| If you...                                          | Use this model           |
| -------------------------------------------------- | ------------------------ |
| Want a fast, interpretable baseline                | GRU or LSTM              |
| Are detecting localized substructures (e.g., -OH)  | 1D CNN                   |
| Have large data and want best performance          | Transformer              |
| Are working with resource-constrained environments | GRU with small embedding |

---

#### 5. Looking Ahead

In Chapter 4.2, we’ll explore an alternative that completely steps away from SMILES: **Graph Neural Networks (GNNs)**. These models work directly on molecular graphs — atoms as nodes, bonds as edges — and naturally capture chemical topology without needing a string format.

---

## 4.2 Graph Neural Networks
In cheminformatics, molecules are often represented in ways that strip away structural context—such as SMILES strings (linearized text) or tabular descriptors (e.g., LogP, molecular weight). While these formats are convenient for storage or computation, they lose the **graph-based nature of chemical structure**, where atoms are nodes and bonds are edges.

**Graph Neural Networks (GNNs)** are designed to work directly with this native molecular form. Rather than relying on handcrafted features or linearized input, GNNs learn directly from **molecular graphs**, capturing both local atomic environments and global topological patterns. This enables them to model nuanced chemical behavior—like electron delocalization, steric hindrance, or intramolecular interactions—that traditional models may overlook.

From predicting toxicity and solubility to quantum properties and biological activity, GNNs have become a cornerstone of modern molecular machine learning. They offer a powerful and general-purpose framework that reflects how chemists naturally think: in terms of atoms and their connections.

In this chapter, we’ll walk through:

- How molecules are converted into graphs for computation
- The concept of **message passing**—the core engine of a GNN
- Variants of GNN architectures commonly used in chemistry
- A full classification example using **PyTorch Geometric (PyG)**
- Practical considerations, limitations, and tuning strategies

GNNs are not just another model type—they represent a shift toward **structure-aware learning**, and mastering them opens the door to high-performance property prediction across drug discovery, materials science, and beyond.

### 4.2.1 What Makes Graphs Unique in Chemistry

In cheminformatics, a molecule is not merely a sequence or a set of features—it is fundamentally a graph. Atoms form the nodes, and bonds between them define the edges. This structural view captures the true nature of molecular interactions and provides a richer, more expressive representation than SMILES strings or numerical descriptors alone.

---

#### Molecules as Graphs

Every chemical compound can be represented as a graph:

- **Nodes (Vertices):** Represent atoms in the molecule
- **Edges:** Represent covalent bonds between atoms
- **Node Features:** Properties of atoms (e.g., atomic number, degree, hybridization)
- **Edge Features:** Bond-specific information (e.g., bond type, aromaticity, conjugation)

This graph-based structure allows machine learning models to learn directly from the connectivity and chemical context of atoms, rather than relying solely on fixed descriptors or sequence representations.

---

#### Why Not Just Use SMILES or Descriptors?

SMILES strings are linear representations of molecules, and while they preserve important information, they do not explicitly capture the topological structure of the molecule. This can lead to several limitations:

| Representation | Strengths                       | Limitations                                   |
| -------------- | ------------------------------- | --------------------------------------------- |
| SMILES         | Easy to parse, compact          | Misses spatial relationships and interactions |
| Descriptors    | Summarize global properties     | Handcrafted, may lose local nuance            |
| Graphs         | Preserve atom–bond connectivity | More complex to model, but more expressive    |

**In essence:** SMILES tells us how a molecule is written, but graphs reveal how it is connected.

---

#### Graph Representation Formats

To process molecules with Graph Neural Networks (GNNs), we must define their structure in machine-readable graph formats. These commonly include:

- **Adjacency Matrix:** A square matrix where A\[i]\[j] = 1 if atom i is bonded to atom j
- **Edge List:** A list of all bond pairs (e.g., (0,1), (1,2), ...)
- **Feature Matrices:**

  - Node features: A matrix of size \[num\_atoms × num\_node\_features]
  - Edge features: Optional matrix encoding bond types

This allows deep learning frameworks like PyTorch Geometric or DGL to build graph-structured batches from molecular datasets.

---

#### Chemical Intuition Behind Graphs

Consider ethanol (`CCO`). Its SMILES string is short, but its graph reveals:

- A carbon backbone
- A terminal hydroxyl group
- A spatial arrangement that affects polarity and hydrogen bonding

Now compare it to dimethyl ether (`COC`). The SMILES strings are similar, but the molecular graph makes clear that the connectivity—and thus chemical behavior—is quite different.

**This highlights the power of graph-based modeling:** structure dictates function, and graphs best preserve structure.

---

#### Summary

- Molecules are naturally graphs, making GNNs a powerful tool in chemistry.
- Graphs preserve atom–bond relationships, offering a richer representation than SMILES or numerical descriptors alone.
- Using node and edge features, graph-based learning enables detailed modeling of molecular interactions.
- GNNs are especially effective in capturing localized effects (e.g., resonance, electronegativity) and global topological properties (e.g., ring systems, branching).

**Next:** We will explore how GNNs perform message passing, the core operation that enables molecules to “communicate” chemically through their atoms and bonds.

---

### 4.2.2 Node and Edge Features in Molecular Graphs

To make accurate predictions, Graph Neural Networks (GNNs) require chemically meaningful input features that describe both the nodes (atoms) and edges (bonds) in a molecular graph. These features serve as the initial state of the graph before any message passing begins. In this section, we’ll explore how chemists translate raw molecular structures into machine-readable formats suitable for GNNs.

---

#### What Are Node and Edge Features?

In a molecular graph:

* Nodes represent atoms
* Edges represent covalent bonds between atoms

Each node and edge must be associated with a feature vector — a numerical encoding that captures local chemical information. These feature vectors are the primary inputs to the GNN and are updated through message passing across the graph.

---

#### Node (Atom) Features

Atom-level features provide local structural and electronic information about each atom. Common node features include:

| Feature        | Description                                     |
| -------------- | ----------------------------------------------- |
| Atomic number  | Integer code for atom type (e.g., C = 6, O = 8) |
| Atom type      | One-hot encoding of atom symbol (e.g., C, N, O) |
| Degree         | Number of directly bonded neighbors             |
| Formal charge  | Net charge of the atom                          |
| Hybridization  | sp, sp2, sp3, etc.                              |
| Aromaticity    | Boolean indicating aromaticity                  |
| Chirality      | R/S stereocenter configuration                  |
| Hydrogen count | Number of implicit and explicit hydrogens       |
| In-ring status | Boolean indicating if atom is in a ring         |

These features are typically extracted using RDKit, which converts SMILES strings into molecular graphs and computes per-atom properties.

---

#### Edge (Bond) Features

Edge features describe how atoms are connected and include:

| Feature              | Description                                   |
| -------------------- | --------------------------------------------- |
| Bond type            | Single, double, triple, aromatic (one-hot)    |
| Conjugation          | Boolean: is bond part of a conjugated system? |
| Ring status          | Boolean: is the bond part of a ring?          |
| Stereo configuration | E/Z (cis/trans) stereochemistry               |

These features are essential for modeling electronic effects, resonance, and steric interactions—factors that often drive chemical reactivity and bioactivity.

---

#### Example: Ethanol (SMILES: CCO)

Let’s consider the molecule ethanol:

* **Atoms:**

  * Carbon 1 (methyl): atomic number = 6, degree = 1
  * Carbon 2 (central): atomic number = 6, degree = 2
  * Oxygen: atomic number = 8, degree = 1

* **Bonds:**

  * C–C: single bond
  * C–O: single bond

A simplified encoding:

* Node features: `[1, 0, 0]` for C, `[0, 1, 0]` for O (example: one-hot atom types)
* Edge features: `[1, 0, 0, 0]` for single bond (one-hot encoding of bond type)

These vectors provide chemically relevant information to the GNN before any learning begins.

---

#### RDKit: Extracting Features from SMILES

We can use RDKit to extract atom and bond features from a SMILES string:

```python
from rdkit import Chem

smiles = "CCO"
mol = Chem.MolFromSmiles(smiles)

# Atom-level features
for atom in mol.GetAtoms():
    print(f"Atom: {atom.GetSymbol()}")
    print(f" - Atomic Num: {atom.GetAtomicNum()}")
    print(f" - Degree: {atom.GetDegree()}")
    print(f" - Is Aromatic: {atom.GetIsAromatic()}")

# Bond-level features
for bond in mol.GetBonds():
    print(f"Bond: {bond.GetBeginAtomIdx()}–{bond.GetEndAtomIdx()}")
    print(f" - Type: {bond.GetBondType()}")
    print(f" - Is Conjugated: {bond.GetIsConjugated()}")
```

This will output per-atom and per-bond features such as atomic number, bond type, and aromaticity—information that can be encoded into vectors for use in a GNN.

---

#### Summary

* Node and edge features form the input layer of any graph neural network applied to molecular data. Without them, the GNN would have no chemically meaningful context to work from.
* Node features capture atomic properties like type, charge, and hybridization.
* Edge features describe bond types and connectivity.
* These features are combined and iteratively refined through message passing.

**Next:** In the next section (4.2.3), we will use these extracted features to build a molecular graph using PyTorch Geometric and begin constructing a complete GNN pipeline.

---

### 4.2.3 Constructing Molecular Graphs with PyTorch Geometric

**Completed and Compiled Code:** *Fully runnable in Google Colab*

Before we can apply a Graph Neural Network (GNN) to a molecule, we need to convert its SMILES string into a graph representation that the model can understand. This includes defining the nodes (atoms), the edges (bonds), and their associated features. In this section, we’ll use the Python library PyTorch Geometric (PyG) to build molecular graphs from SMILES using features extracted with RDKit.

---

#### Overview of the Workflow

* Parse SMILES using RDKit to extract the molecular structure
* Define node features: For each atom, compute a feature vector (e.g., atomic number, degree, aromaticity)
* Define edge index: List all bonds as pairs of connected atoms
* Define edge features: For each bond, compute features like bond type and conjugation
* Package into a `torch_geometric.data.Data` object — the standard graph container in PyG

---

#### Installation (Google Colab)

```python
# PyTorch and PyTorch Geometric setup (Colab)
!pip install -q rdkit
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install -q torch-geometric
```

---

#### Example: Convert a Single SMILES to a PyG Graph

```python
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Data

# Helper function to get atom features
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        int(atom.GetIsAromatic())
    ], dtype=torch.float)

# Helper function to get bond features
def bond_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated())
    ], dtype=torch.float)

# Convert SMILES to molecular graph
def smiles_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_feats)  # Shape: [num_nodes, num_node_features]

    # Edge list and edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # undirected edges
        edge_feat = bond_features(bond)
        edge_attr += [edge_feat, edge_feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Test with a molecule
graph = smiles_to_pyg("CCO")  # Ethanol
print(graph)
```

---

#### Output (Graph Summary)

```text
Data(x=[3, 3], edge_index=[2, 4], edge_attr=[4, 5])
```

**Explanation:**

* `x=[3, 3]`: 3 atoms with 3 features each
* `edge_index=[2, 4]`: 4 directed edges (2 bonds, bidirectional)
* `edge_attr=[4, 5]`: 4 edges with 5-dimensional bond features

---

#### Feature Explanation

* **Node Features (x):** `[Atomic Number, Degree, Is Aromatic]`

  * e.g., `[6, 4, 0]` for a non-aromatic carbon with 4 neighbors
* **Edge Features (edge\_attr):** `[is_single, is_double, is_triple, is_aromatic, is_conjugated]`

  * e.g., `[1, 0, 0, 0, 0]` for a plain single bond

---

#### Practice Problem 1: Visualizing Molecular Graph Features

**Task:**

* Use RDKit to parse a molecule of your choice
* Extract and print atom and bond features using the `smiles_to_pyg()` function
* Try SMILES like: `"c1ccccc1O"` (phenol) or `"CC(=O)O"` (acetic acid)

```python
your_smiles = "CC(=O)O"
graph = smiles_to_pyg(your_smiles)

print("Node Features:")
print(graph.x)

print("\nEdge Index:")
print(graph.edge_index)

print("\nEdge Features:")
print(graph.edge_attr)
```

---

#### Summary

* Parse a SMILES string into a graph of atoms and bonds
* Extract chemically meaningful node and edge features
* Format the molecule as a PyTorch Geometric `Data` object

**Next:** In the next section (4.2.4), we’ll use these graph objects to build and train a real GCN model for molecular property prediction.

---

### 4.2.3 Constructing Molecular Graphs with PyTorch Geometric

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/1zQq6MJU6Al4QiV309mzXRi5DNB3o5Tko?usp=sharing)

Before we can apply a Graph Neural Network (GNN) to a molecule, we need to convert its SMILES string into a graph representation that the model can understand. This includes defining the nodes (atoms), the edges (bonds), and their associated features. In this section, we’ll use the Python library PyTorch Geometric (PyG) to build molecular graphs from SMILES using features extracted with RDKit.

---

#### Overview of the Workflow

* Parse SMILES using RDKit to extract the molecular structure
* Define node features: For each atom, compute a feature vector (e.g., atomic number, degree, aromaticity)
* Define edge index: List all bonds as pairs of connected atoms
* Define edge features: For each bond, compute features like bond type and conjugation
* Package into a `torch_geometric.data.Data` object — the standard graph container in PyG

---

#### Installation (Google Colab)

```python
# PyTorch and PyTorch Geometric setup (Colab)
!pip install -q rdkit
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install -q torch-geometric
```

---

#### Example: Convert a Single SMILES to a PyG Graph

```python
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Data

# Helper function to get atom features
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        int(atom.GetIsAromatic())
    ], dtype=torch.float)

# Helper function to get bond features
def bond_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated())
    ], dtype=torch.float)

# Convert SMILES to molecular graph
def smiles_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_feats)  # Shape: [num_nodes, num_node_features]

    # Edge list and edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # undirected edges
        edge_feat = bond_features(bond)
        edge_attr += [edge_feat, edge_feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Test with a molecule
graph = smiles_to_pyg("CCO")  # Ethanol
print(graph)
```

---

#### Output (Graph Summary)

```text
Data(x=[3, 3], edge_index=[2, 4], edge_attr=[4, 5])
```

**Explanation:**

* `x=[3, 3]`: 3 atoms with 3 features each
* `edge_index=[2, 4]`: 4 directed edges (2 bonds, bidirectional)
* `edge_attr=[4, 5]`: 4 edges with 5-dimensional bond features

---

#### Feature Explanation

* **Node Features (x):** `[Atomic Number, Degree, Is Aromatic]`

  * e.g., `[6, 4, 0]` for a non-aromatic carbon with 4 neighbors
* **Edge Features (edge\_attr):** `[is_single, is_double, is_triple, is_aromatic, is_conjugated]`

  * e.g., `[1, 0, 0, 0, 0]` for a plain single bond

---

#### Practice Problem 1: Visualizing Molecular Graph Features


**Task:**

* Use RDKit to parse a molecule of your choice
* Extract and print atom and bond features using the `smiles_to_pyg()` function
* Try SMILES like: `"c1ccccc1O"` (phenol) or `"CC(=O)O"` (acetic acid)


Practice Problem 1 Solution: Visualizing Molecular Graph Features

```python
# Practice Problem 1: Visualizing features for acetic acid
your_smiles = "CC(=O)O"  # Acetic acid

graph = smiles_to_pyg(your_smiles)

print("Node Features (x):")
print(graph.x)

print("
Edge Index (edge_index):")
print(graph.edge_index)

print("
Edge Features (edge_attr):")
print(graph.edge_attr)
```

**Sample Output:**

```text
Node Features (x):
tensor([[6., 4., 0.],  # Carbon atom (methyl)
        [6., 3., 0.],  # Carbon atom (carbonyl)
        [8., 1., 0.]]) # Oxygen atom (hydroxyl)

Edge Index (edge_index):
tensor([[0, 1, 1, 2],
        [1, 0, 2, 1]])

Edge Features (edge_attr):
tensor([[1., 0., 0., 0., 0.],  # C–C single bond
        [1., 0., 0., 0., 0.],  # C–C (reverse)
        [0., 1., 0., 0., 0.],  # C=O double bond
        [0., 1., 0., 0., 0.]]) # C=O (reverse)
```

---

#### Analysis

This problem helps solidify how molecules are translated into graph structures with chemically meaningful features:

* **Nodes** represent atoms, each with a 3-element vector:

  * Atomic number (e.g., 6 = carbon, 8 = oxygen)
  * Degree (number of bonds)
  * Aromaticity (all 0 here because acetic acid is non-aromatic)

* **Edges** represent bonds, and each bond appears twice (once for each direction) in `edge_index`. Their associated `edge_attr` vectors show:

  * The first bond is a single bond → `[1, 0, 0, 0, 0]`
  * The second bond is a double bond → `[0, 1, 0, 0, 0]`

This illustrates how even simple molecules like acetic acid are encoded with precise structural features, setting the stage for GNNs to learn from both atomic identity and bonding context.

**Why it matters:**

This feature extraction step is not just a formality. It gives the GNN everything it needs to begin learning — atomic types, bonding patterns, and spatial context.

Without this, message passing would be blind to chemical reality.

---

#### Summary

* Parse a SMILES string into a graph of atoms and bonds
* Extract chemically meaningful node and edge features
* Format the molecule as a PyTorch Geometric `Data` object

**Next:** In the next section (4.2.4), we’ll use these graph objects to build and train a real GCN model for molecular property prediction.

---

### 4.2.4 Training a GNN Model for Molecular Property Prediction

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/19uFxeMvDbW3MxJ-UV6zKL5cf-axX0IKG?usp=sharing)

Once we’ve converted SMILES into graph-based data structures, we’re ready to train a Graph Neural Network (GNN) to predict molecular properties. In this section, we’ll demonstrate how to train a Graph Convolutional Network (GCN) to classify blood–brain barrier permeability using the BBBP dataset, building directly on the graph representations discussed in 4.2.3.

We’ll use PyTorch Geometric (PyG), a popular framework for building and training GNNs.

---

#### Problem Setup: BBBP Classification with GCN

* **Objective:** Train a GCN to classify molecules as either permeable (1) or non-permeable (0) to the blood–brain barrier.
* **Input:** Molecular graphs constructed from SMILES (nodes = atoms, edges = bonds)
* **Output:** A single prediction per graph indicating BBB permeability

---

#### Code: GCN Training Pipeline

This code can be run in Google Colab (with GPU acceleration enabled):

```python
# Step 1: Install PyTorch Geometric
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install -q torch-geometric
```

```python
# Step 2: Load and preprocess the dataset
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import one_hot
from torch_geometric.loader import DataLoader

# Node and bond feature helpers
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        int(atom.GetIsAromatic())
    ], dtype=torch.float)

def bond_features(bond):
    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }
    btype = bond_types.get(bond.GetBondType(), 4)
    return one_hot(torch.tensor(btype), num_classes=5).float()

def smiles_to_data(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.float))

# Step 3: Load BBBP and convert to graphs
url = "https://raw.githubusercontent.com/Data-Chemist-Handbook/Data-Chemist-Handbook.github.io/refs/heads/master/_pages/BBBP.csv"
df = pd.read_csv(url)

graph_list = [smiles_to_data(smi, lbl) for smi, lbl in zip(df['smiles'], df['p_np'])]
graph_list = [g for g in graph_list if g is not None]

# Step 4: Split and load data
train_data, test_data = train_test_split(graph_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)
```

---

#### Step 5: Define the GCN Model

```python
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))
```

---

#### Step 6: Train and Evaluate the Model

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNModel().to(device)

import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train():
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch).squeeze()
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()

# Evaluation loop
def test():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch).squeeze()
            pred = (output > 0.5).float()
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

# Run training
for epoch in range(5):
    train()
    acc = test()
    print(f"Epoch {epoch+1}, Test Accuracy: {acc:.2f}")
```

---

#### Example Output

```text
Epoch 1, Test Accuracy: 0.78  
Epoch 2, Test Accuracy: 0.81  
Epoch 3, Test Accuracy: 0.83  
Epoch 4, Test Accuracy: 0.84  
Epoch 5, Test Accuracy: 0.86
```

---

#### Analysis

This model achieves strong performance (>85% accuracy) after just a few epochs — comparable to RNN and feedforward baselines — but using raw molecular structure in graph form.

* GCN Layers allow the model to combine information across atoms and bonds via message passing.
* Global Pooling compresses node-level information into a fixed-size graph-level feature vector.
* Binary Cross-Entropy Loss measures how well the model distinguishes permeable vs. non-permeable molecules.

**Key Advantages:**

* Unlike SMILES-based models, GNNs are inherently aware of molecular topology.
* Node and edge features encode rich chemical semantics, improving generalization to unseen molecules.

---

### 4.2.5 Interpreting GNN Predictions and Attention Weights

After training a Graph Neural Network (GNN) to predict molecular properties, an important question arises: how can we understand the reasoning behind the model’s predictions? In scientific applications like cheminformatics, interpretability is not merely a luxury—it is a necessity for evaluating reliability, guiding experimental follow-up, and generating new chemical insights.

In this section, we explore techniques for interpreting GNN models, with a focus on attention-based GNNs, which provide a natural framework for identifying which atoms and bonds contributed most to a molecular prediction.

---

#### Importance of Interpretability in Molecular Modeling

In many chemical applications, the prediction alone is not sufficient. Chemists often require answers to the following types of questions:

* Which atoms or substructures caused the model to assign a high toxicity score?
* What chemical features led the model to predict high solubility or permeability?
* Is the model focusing on chemically meaningful patterns or overfitting to irrelevant noise?

These questions are critical in:

* Mechanistic understanding (e.g., identifying functional groups associated with biological activity or adverse effects)
* Lead optimization (e.g., refining specific substructures to improve pharmacokinetic properties)
* Model debugging (e.g., detecting spurious correlations in the data)

Without interpretability, model predictions remain black-box outputs, limiting their usefulness in research and decision-making.

---

#### Attention Mechanisms in GNNs

Graph Attention Networks (GATs) extend traditional GNNs by incorporating an attention mechanism during the message passing phase. In standard message passing, all neighboring nodes contribute equally to the update of a node’s feature vector. In contrast, GATs learn attention weights that quantify the relative importance of each neighbor when updating node representations.

Mathematically, the attention coefficient $\alpha_{ij}$ quantifies how much node $j$'s information contributes to node $i$'s updated representation. This coefficient is computed as:

$\alpha_{ij} = \text{softmax}_j(\mathbf{a}^T \cdot \text{LeakyReLU}(\mathbf{W}[\mathbf{h}_i \| \mathbf{h}_j]))$

Where:

* $\mathbf{h}_i$ and $\mathbf{h}_j$ are the feature vectors of nodes $i$ and $j$
* $\mathbf{W}$ is a learnable linear transformation
* $\mathbf{a}$ is a learnable attention vector
* $\|$ denotes vector concatenation
* The softmax ensures attention scores are normalized across neighbors

These learned coefficients allow the network to focus more on chemically important atoms during graph aggregation.

---

#### Interpreting Attention Scores in Chemical Graphs

In the context of molecular graphs, attention scores provide a way to assess which atoms and bonds the model considered most important when making its prediction. By visualizing these scores, chemists can gain insight into which structural elements were most influential. For instance:

* In toxicity prediction, the model may assign high attention to nitro groups, halogens, or specific ring systems.
* In permeability modeling, attention may concentrate around hydrophobic regions or polar groups, depending on their influence on blood–brain barrier penetration.

These attention scores are particularly useful in identifying structure–activity relationships (SARs) that are not explicitly encoded in the input features but are learned during training.

---

#### Code Example: Extracting Attention Scores in PyTorch Geometric

```python
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=1)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1)

    def forward(self, x, edge_index):
        x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        return x, attn1, attn2
```

The `return_attention_weights=True` argument causes each GATConv layer to return not only the transformed node features, but also the attention weights. These can then be visualized using molecular graph plotting libraries (e.g., RDKit, NetworkX) to highlight atom–atom interactions based on their importance.

---

#### Example: Interpreting Attention in a BBB Permeability Task

Suppose a trained GAT model predicts that a given molecule is BBB-permeable with 92% confidence. By extracting and visualizing the attention weights:

* The model may focus on a primary amine and an aromatic ring, indicating that these fragments contributed most strongly to the prediction.

This aligns with medicinal chemistry intuition—aromatic groups and basic amines are known to facilitate CNS penetration through passive diffusion or transporter affinity.

Such interpretability strengthens the chemist's trust in the model and may guide subsequent molecular design.

---

#### Summary

* Graph Attention Networks provide a built-in mechanism for interpreting GNN decisions through attention weights.
* These weights can be used to identify chemically meaningful substructures that influence molecular property predictions.
* Attention visualization supports hypothesis generation, SAR analysis, and trust in model-guided workflows.
* In chemical applications, model transparency is often as important as accuracy—especially when decisions must be validated experimentally.

**Next:** We will move from interpretation back to application: using full GNN pipelines to predict molecular properties from graph-structured data.

---

### 4.2.6 Full GNN Pipeline for Molecular Property Prediction

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/11XwnJbZytO32Bmb4t6KTwyLdmdaiKegr?usp=sharing)

After exploring graph representations, feature construction, message passing, and interpretability, it’s time to put everything together into a complete end-to-end Graph Neural Network (GNN) pipeline. In this section, we’ll walk through the full process of training a GNN to predict a molecular property—in this case, blood–brain barrier permeability (BBBP)—directly from molecular graphs derived from SMILES strings.

This will demonstrate how to convert molecules into graph objects, encode atomic and bond features, construct and train a GNN, and evaluate its predictive performance.

---

#### Overview of the GNN Pipeline

The molecular GNN workflow typically involves the following steps:

* **Data Acquisition**

  * Load a dataset of molecules with labeled properties (e.g., BBBP dataset).

* **Graph Construction**

  * Convert each molecule’s SMILES string into a graph using RDKit and build `torch_geometric.data.Data` objects for PyTorch Geometric.

* **Feature Engineering**

  * Encode node features (atom-level) and edge features (bond-level).

* **Model Design**

  * Define a graph-based neural network (e.g., GCN, GAT, MPNN).

* **Training and Evaluation**

  * Train the model to minimize prediction error, and assess accuracy on unseen test molecules.

---

#### Code Example: BBBP Prediction with GCN

```python
# Step 1: Install required libraries
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install torch-geometric
!pip install rdkit!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
!pip install torch-geometric
!pip install rdkit
```

```python
# Step 1: Install required libraries
import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


url = "https://raw.githubusercontent.com/Data-Chemist-Handbook/Data-Chemist-Handbook.github.io/refs/heads/master/_pages/BBBP.csv"
df = pd.read_csv(url)

# Step 3: Create atom features
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())
    ]

# Step 4: Build PyTorch Geometric Data objects
molecules = []
for i, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['smiles'])
    if mol is None:
        continue

    atoms = mol.GetAtoms()
    atom_feats = [atom_features(atom) for atom in atoms]
    
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # undirected
    
    x = torch.tensor(atom_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([row['p_np']], dtype=torch.float)

    molecules.append(Data(x=x, edge_index=edge_index, y=y))

# Step 5: Train/test split
train_data, test_data = train_test_split(molecules, test_size=0.2, random_state=42)
```

---

#### Building the GCN Model

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

class GCNClassifier(torch.nn.Module):
    def __init__(self):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(4, 32)
        self.conv2 = GCNConv(32, 64)
        self.linear = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.linear(x)).squeeze()
```

---

#### Training the GCN

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
```

---

#### Evaluating Performance

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.batch) > 0.5
        correct += (preds == batch.y.bool()).sum().item()
        total += batch.y.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2f}")
```

---

#### Results and Analysis

In this pipeline, we built a GCN that achieved reasonable classification accuracy on the BBBP dataset using just four simple atom-level features. While this is a minimal setup, it demonstrates the power of GNNs to model molecular structure and predict pharmacological properties.

* The GCN layers aggregate information across the molecular graph.
* The global pooling layer summarizes atom-level signals into a single molecular fingerprint.
* The final sigmoid layer estimates the probability of blood–brain barrier permeability.

This basic model is highly extensible. By incorporating richer features (e.g., additional atom/bond descriptors, edge weights), stacking more layers, or using attention mechanisms, we can improve performance and extract deeper insights.

---

#### Key Takeaways

* GNNs can be applied end-to-end for molecular property prediction from SMILES.
* PyTorch Geometric enables scalable graph processing with minimal boilerplate code.
* Even simple GCNs can learn structure–activity relationships effectively.
* This pipeline is a foundation for more advanced architectures such as MPNNs or attention-based models.

**Next:** We’ll turn to Random Forests to revisit traditional ensemble models and compare their strengths and weaknesses relative to GNNs and deep learning approaches.

---

## 4.3 Random Forests

## 4.4 Neural Networks
