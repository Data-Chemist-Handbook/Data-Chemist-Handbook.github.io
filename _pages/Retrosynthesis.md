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

Another limitation of SMILES-based methods is their inability to effectively capture molecules' structural information such as atomic properties, bond features, and adjacency relationships. In addition, current models struggle to fully exploit the potential of multiple molecular descriptors. This challenge often necessitates trade-offs between computational efficiency and predictive accuracy. For example, molecular fingerprinting emphasizes detailed structural features, while SMILES provides more global molecular information. When using only one descriptor, important molecular characteristics may be lost.

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

**Complete code:** [Click here](https://colab.research.google.com/drive/1s0l_0kBmZCmnXqwTekuPG3Zn2kVpOMNu?usp=sharing)

**Dataset:** The USPTO-50k dataset was used for this model. Link: [Click here](https://figshare.com/articles/dataset/USPTO-50K_raw_/25459573?file=45206101)

**Step 1: Download the data files and upload them to Colab**

The provided link has downloadable raw files split into `raw_train.csv`, `raw_val.csv`, and `raw_test.csv`. Download the zip, extract files, and upload into the Colab notebook.

**Step 2: Install and Import Required Libraries**

```python
# Install RDKit, transformers and any other missing libraries in Google Colab
!pip install rdkit
!pip install transformers
!pip install wandb # Optional: Used for hyperparameter tuning or logging models

import pandas as pd
from rdkit import Chem
from transformers import RobertaTokenizerFast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
```
**Hyperparameter tuning**:

***Note:*** *To use transformers library, add your HuggingFace token to the Colab notebook. The HuggingFace token can be found in "Settings>>Access Tokens" when logged in to HuggingFace (more information [here](https://huggingface.co/docs/hub/en/security-tokens)). To add the key to Colab, click on the key icon on the left side panel of the notebook and paste the token in the value field. Name the token "HF_TOKEN" and toggle notebook access for the key.* 

*If not using Colab, the following lines of code can be used to access the token:*
```python
# If using Colab, add HF_TOKEN to Colab Secrets
# else, add HF_TOKEN to python env and use the following lines of code
import os
access_token = os.environ.get('HF_TOKEN')
```

**Step 3: Data loading and Processing**

```python
# Data loading and Processing
# Paths to data files
train_file = "raw_train.csv"
val_file = "raw_val.csv"
test_file = "raw_test.csv"

# Load the data
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

# File structure: "ID, Class, reactants>>products"
# Extract only the third column: "reactant>>products"
train_rxns = train_df.iloc[:, 2]
val_rxns = val_df.iloc[:, 2]
test_rxns = test_df.iloc[:, 2]

# Sanity check
print(train_rxns.head())
print(val_rxns.head())
print(test_rxns.head())
```

**Step 4: Separate reactants and products**

```python
# Separate reactants and products
def extract_pairs(reaction_series):
    inputs = []
    outputs = []
    for rxn in reaction_series:
        try:
            reactants, products = rxn.split(">>")
            inputs.append(products.strip())
            outputs.append(reactants.strip())
        except ValueError:
            continue  # skip malformed lines
    return inputs, outputs

train_X, train_y = extract_pairs(train_rxns)
val_X, val_y = extract_pairs(val_rxns)
test_X, test_y = extract_pairs(test_rxns)

# Sanity check
print(train_y[0]+">>"+train_X[0]==train_rxns[0])
```

**Step 5: Canonicalize SMILES using RDKit**

With SMILES represntation, a single molecule can be represnted by more than one valid SMILES string. This means that the same molecule could appear multiple times in the dataset with different SMILES and models might overfit or mislearn due to inconsistent representations. To solve this issue, we use SMILES canonicalization, which converts different valid SMILES strings that represent the same molecule into a unique, standardized form (called canonical SMILES).

```python
# Camonicalise SMILES
def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def canonicalize_pairs(X, y):
    canon_X, canon_y = [], []
    for prod, react in zip(X, y):
        # Canonicalize product
        canon_prod_parts = []
        for p in prod.split('.'):
            c = canonicalize(p)
            if c:
                canon_prod_parts.append(c)
        c_prod = '.'.join(canon_prod_parts)
        # Canonicalize reactants
        canon_react_parts = []
        for r in react.split('.'):
            c = canonicalize(r)
            if c:
                canon_react_parts.append(c)
        c_react = '.'.join(canon_react_parts)
        if c_prod and c_react:
            canon_X.append(c_prod)
            canon_y.append(c_react)
    return canon_X, canon_y

train_X, train_y = canonicalize_pairs(train_X, train_y)
val_X, val_y = canonicalize_pairs(val_X, val_y)
test_X, test_y = canonicalize_pairs(test_X, test_y)

# Sanity check
print(train_X[0])
```

**Step 6: Tokenize SMILES**

```python
# Tokenize SMILES

tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k") # Or use other tokenizer of choice

# Create function for tokenization
def tokenize_smiles_bpe(smiles_list, tokenizer, max_length=600):
    encodings = tokenizer(smiles_list,
                          padding='max_length',
                          truncation=True,
                          max_length=max_length,
                          return_tensors='pt')
    return encodings['input_ids'] #, encodings['attention_mask'] use for transformer maybe

# Tokenize encoder (product SMILES)
train_enc_input = tokenize_smiles_bpe(train_X, tokenizer)
val_enc_input   = tokenize_smiles_bpe(val_X, tokenizer)
test_enc_input  = tokenize_smiles_bpe(test_X, tokenizer)

# Tokenize decoder (reactant SMILES)
train_dec_input = tokenize_smiles_bpe(train_y, tokenizer)
val_dec_input   = tokenize_smiles_bpe(val_y, tokenizer)
test_dec_input  = tokenize_smiles_bpe(test_y, tokenizer)

# Sanity check
print(train_enc_input.shape)
print(train_dec_input.shape)
```

**Step 7: Define Some Helpful Helpers**

`create_dataloader` is a utility function used to wrap input and target tensors into a DataLoader object which handles batching, shuffling etc. Additionally, this function ensures that both enc_inputs and dec_inputs are PyTorch tensors. If they're not already tensors, it converts them.

```python
# Dataset wrapper
def create_dataloader(enc_inputs, dec_inputs, batch_size):
    inputs = enc_inputs if isinstance(enc_inputs, torch.Tensor) else torch.tensor(enc_inputs, dtype=torch.long)
    targets = dec_inputs if isinstance(dec_inputs, torch.Tensor) else torch.tensor(dec_inputs, dtype=torch.long)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

Next, we define the `compute_accuracy` function which calculates token-level accuracy of predictions from the model, while ignoring padded positions. Token-level accuracy is used here during the training and evaluation loops as sequence-level comparisions may be too harsh for the model to learn. During testing, sequence-level checking may be used e.g., exact match or top-k accuracy.

```python
# Helper function
def compute_accuracy(predictions, targets, pad_token_id=0):
    preds = predictions.argmax(dim=2)  # shape: (batch_size, seq_len)
    mask = targets != pad_token_id     # ignore padding
    correct = (preds == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()
```

We also define `device` to define the appropriate device to run the model, specifically GPU or CPU.

```python
# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Finally, we may optionally use `wandb` to tune hyperparameters during training by setting several possible ranges for these as seen in the following code. Note that the sweep values shown are simply for demonstration purposes.

```python
# WandB Sweep Options Configuration (if tuning hyperparameters)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'distribution': 'uniform', 'min':0.0001, 'max': 0.01},
        'embed_dim': {'values': [128, 256]},
        'hidden_dim': {'values': [ 256, 512]},
        'num_layers': {'values': [2,3,4]},
        'dropout': {'distribution': 'uniform', 'min':0.2, 'max': 0.7},
        'epochs': {'distribution': 'int_uniform', 'min': 2, 'max': 25},
    }
}
```


## 7.4 Transformer

## 7.5 Graph Neural Networks
