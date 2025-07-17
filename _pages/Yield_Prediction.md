---
title: 8. Yield Prediction
author: Haomin-QuangDao
date: 2024-08-18
category: Jekyll
layout: post
---

**Chemical Reaction Yield**: In any chemical reaction, the yield refers to the percentage of the reactants that is succesfully converted to desired product. For chemists, predicting this yield is crucial because a high predicted yield can save time and resources by guiding experiment planning, while a low yield might indicate an inefficient route Traditionally, chemists have used domain knowledge or trial-and-error to
estimate yields, but modern machine learning methods can learn yield patterns from data and make fast predictions.  

**Machine Learning Mode**:Predicting reaction yield can be approached with a variety of machine learning models, ranging from
classical ensemble methods to modern deep neural networks. In this section, we discuss how Recurrent Neural Networks (RNNs), Graph Neural Networks (GNNs), Random Forests, and Neural Networks have been applied to chemical yield prediction. For each model type, we explain its fundamental working principle in accessible terms, why it suits yield prediction, give chemistry-focused
examples (including datasets like reaction screens and patent data), and provide simple code Python examples. We also compare typical performance metrics reported on benchmark reaction yield datasets.

---

## 8.1 Recurrent Neural Networks (RNNs)

### Why using RNN ?
Early data-driven yield prediction models often used fixed descriptors or one-hot encodings of reactants (e.g. encoding presence/absence of certain functional groups) and then applied algorithms like Random Forests. While useful, those approaches treat
each reaction as a bag of features, potentially missing the structure of the molecular and reaction conditions influence yield. Recent advances bring a big advancement into yield prediction field which is deep learning model. In particular, treating a reaction’s representation as a sequence – analogous to a sentence in a chemical “language” – has proven a high potential. For example, a reaction can be written as a SMILES string (a text encoding of molecules), and deep learning models can be trained to read this string and predict the yield . One way to do this is with Recurrent Neural Networks (RNNs), which are designed to handle sequential data.  

### What is RNN actually ?
Imagine reading a recipe for a chemical reaction step by step. An RNN does something similar: it reads a
sequence (like a reaction encoded as text) one element at a time, remembering what came before. This
memory is what we call the hidden state. At each step in the sequence, the RNN updates its hidden state
based on the current input and the previous state, and optionally produces an output (in our case,
eventually an estimated yield). Figuratively, you can think of the hidden state as the RNN’s interpretation of
the reaction so far.  
In a chemistry context, consider an RNN reading a reaction SMILES string token by token. When it sees the “Br” token (aryl bromide), it updates its hidden state to reflect the presence of a highly reactive aryl halide (which often increases yield). Later, encountering a token for a bulky base, it further adjusts the hidden state to represent potential steric hindrance (which can lower yield). In this way, the RNN builds a running summary of the reaction, step by step incorporating each component’s influence on the final yield.  

#### Key characteristics of RNNs  
- **Sequence Awareness:** RNNs process input sequences in order, which is useful in chemistry because
the order of components (and the context they appear in) can matter. For example, an RNN can learn
that a brominated reactan at the start of a SMILES might indicate a different reactivity pattern than chlorinated due to bromine being a better leaving group.  
- **Hidden State (Memory):** As the RNN reads through the sequence, it carries a hidden state vector.
This is like the running summary of what has been “seen” so far. In chemistry, you can compare it to
a chemist keeping track of which functional groups have appeared in a reaction and adjusting
expectations of yield accordingly.  
- **Reusability of Structure:** The RNN uses the same network cell for each step (recurrently). This is
analogous to applying the same reasoning rules to each part of a molecule or reaction. No matter
where a particular functional group appears in the SMILES, the RNN cell can recognize it and update
the state in a consistent way.  

![RNN Diagram](../../resource/img/yield_prediction/RNN_Process.png)
Figure 1: A schematic “unrolling” of the RNN over three time steps as it reads a reaction sequence token-by-token. At each step t, the input token 𝑥_𝑡 (e.g., “Br”, “Ph”, “R”) enters an RNN cell. That cell combines the new input with the previous hidden state hₜ₋₁ to produce the updated hidden state hₜ. Arrows show the flow: horizontal arrows carry the hidden state forward through time, and vertical arrows inject each new chemistry token into the cell. This illustrates how the network incrementally builds a memory of the reaction’s components, accumulating context that will ultimately be used to predict the reaction’s yield.  

Chemically speaking, an RNN can learn patterns by seeing many examples. It remembers the
presence of the previous states while reading the rest of the reaction. This ability to capture context and sequence is
what sets RNNs apart from simpler models in yield prediction.  

### RNN Architecture for Yield Prediction
To understand how an RNN predicts yields, let’s break down its architecture into parts, relating each to a
chemistry analogy:  
- **Input Layer (Encoding the Reaction):** First, we need to encode the reaction (e.g., as a SMILES
string). This could be done at the character level or using chemical tokens. For simplicity, imagine we
feed the SMILES one character at a time into the RNN. We typically convert characters to numeric
vectors via an embedding layer so it is essentially a lookup table that turns each symbol (like C , Br ,
= , . etc.) into a vector of numbers. This is analogous to encoding chemical species properties:
just as we might encode an element or functional group with descriptors, here we learn a vector
representation for each character or token.
- **Recurrent Layer (the “RNN cell”):** This is the heart of the network. A basic RNN cell takes two things
as input: the current input vector (from the reaction sequence) and the previous hidden state. It
combines them (through weighted summation and a nonlinear function) to produce a new hidden
state. Mathematically, one common formulation is: **hₜ = tanh(W ⋅ xₜ + U ⋅ hₜ₋₁ + b)** where xₜ is the input at step t, hₜ₋₁ is the hidden state from the previous step, W, U and b are learnable parameters (matrices/vectors) . The new hidden state is then passed to the next step. You can picture this like an assembly line: each station (time step) takes the current reagent
(input) and the accumulated mixture info (hidden state) to update the mixture’s status.  
- **Output Layer:**: After processing the entire sequence (the RNN has seen the whole reaction), we use
the final hidden state (or some aggregation of all hidden states) to predict the yield value. Typically,
a simple fully-connected layer takes the last hidden state and outputs a number (which we interpret as the predicted yield). We often apply a suitable activation if needed (for regression usually identity or ensure the output is within 0-100 range by clamping or using a function, though many models just predict a real number and we treat it as yield %).  
- **Longer-term Dependencies – LSTM/GRU:** One issue with basic RNNs is that their memory can fade for long sequences (vanishing gradient problem). In lengthy SMILES or very complex reactions, something seen at the start can be “forgotten” by the end. Advanced RNN variants like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) address this by having gated mechanisms that control information flow. In chemical analogy, an LSTM is like a careful lab notebook. It decides what to write down or erase at each step, so critical information isn’t lost by the time the network finishes reading the reaction. These gated RNNs maintain a more persistent cell state in addition to the hidden state, which helps in capturing long-range effects (e.g., a reagent mentioned at the beginning of a protocol affecting the final outcome). Many state-of-the-art sequence models for chemistry use LSTMs or GRUs under the hood because of these benefits.  

**RNN Diagram**  
![RNN Diagram](../../resource/img/yield_prediction/RNN.png)
A compact schematic of an RNN unrolled over three time steps for yield forecasting. Blue circles (hₜ₋₁, hₜ, hₜ₊₁) are the hidden‐state vectors; at t - 1, the model ingests the reaction condition (orange), updates to hₜ, then outputs the yield at 𝑡. That yield feeds back into the next step, combining with hₜ to form hₜ₊₁, which produces the predicted yield at t+1.  
### How RNNs Learn – The Math Behind the Scenes  
How does an RNN actually learn these chemical yield patterns from data? It all happens during the training
process via backpropagation. Here’s an overview, tied into our yield prediction scenario:
- **1. Training Data:** We provide the RNN with many example reactions and their yields. For instance, a dataset might contain thousands of reactions (as SMILES or similar text) each labeled with an experimentally measured yield (as a percentage) . An example entry might be: “Some Reaction – Yield: 85%”. The model doesn’t know chemistry, but it will try to map from the reaction string to that number.  
- **2. Forward Pass:** For each training reaction, the RNN processes the sequence of tokens, updating its
hidden state, and finally produces a predicted yield $\hat{y}$ (a single number). Initially, these
predictions are basically random, since weights are random at start.  
- **3. Loss Calculation:** We measure the error of the prediction using a loss function. In regression tasks
like yield, a common choice is Mean Squared Error (MSE): $\mathcal{L}$ = ($\hat{y}$ − y)^2 where y is the true yield and $\hat{y}$ is the predicted yield. For example, if the true yield was 85 and the model predicted 60, the error is $(60 - 85)^2 = 625$ – quite large. The model will try to reduce this.
- **4. Backpropagation Through Time (BPTT):** This is where the “magic” happens. The model computes how the loss $\mathcal{L}$ changes with respect to each of the RNN’s parameters. Because the RNN’s output depends on all the steps (all the way back to the first token of the reaction), the error is back-propagated through each time step of the RNN. Essentially, the RNN unrolls through the sequence and perform backpropagation through that unrolled network. During training, the gradient $\displaystyle \frac{\partial \mathcal{L}}{\partial W_{Br}}$ tells us how much the loss changes if we tweak the weight on the “Br” token. For instance,  $\displaystyle \frac{\partial \mathcal{L}}{\partial W_{Br}}$=−0.5, then increasing $W_{Br}$ by a small step (say, 0.1) will reduce the loss increase yield prediction by 5 for this reaction.  
- **5. Parameter Update:** Using an optimization algorithm (like Adam or simple gradient descent), we adjust the weights slightly in the direction that reduces the error. For example, if the network underestimated yields whenever a bromide ( Br ) was present, the gradients will nudge the relevant weights to boost yield predictions in presence of Br. The model learns that bromides often lead to higher yields (compared to chlorides, perhaps) because that adjustment reduces overall prediction error across training examples.  
- **6. Iterate:** The process repeats for many epochs (passes through the data). Over time, the RNN’s hidden state representations become tuned to capture factors that influence yield. It might learn numeric encodings corresponding to “presence of electron donor”, “high temperature used”, “watersensitive reagent present” etc., without being explicitly told these concepts – they emerge from the training if those factors consistently affect yield in the data.  
  
Mathematically, the training is adjusting the matrices like $W, U$ (from the RNN cell) and the output layer
weights to minimize the loss across all training reactions. One can think of the hidden state at final time as
computing some function $f(\text{reaction}) = h_{\text{final}}$, and then the output layer does $y = w^\top
h_{\text{final}} + b$. Training tunes $f$ and $w$ so that $y$ is close to the true yield for all training
reactions.  
  
From a chemistry standpoint, during training the model might discover, for example, that “if nitrogen is
present in the base, yields tend to be lower” because every time a reaction with, say, a nitrogenous
base had lower yield, the model was penalized until it adjusted to predict that lower yield. Similarly, it might
learn “reactions with certain protecting groups never go to completion” and incorporate that into its hidden
state memory.   
  
It’s worth noting that RNNs don’t have built-in knowledge of chemistry; they learn purely from data
correlations. Thus, they require sufficient examples of various factors to generalize well. If an RNN is trained
on a high-throughput experimentation (HTE) dataset (like the Buchwald-Hartwig amination yields from Doyle’s group ), it can achieve very high accuracy on similar reactions. However, if it is asked a very different chemistry than it saw in training, it might falter. Just like humans, we tend to perform poorly on tasks we haven’t learned before.  

**Training an RNN:** summary of steps:  
- Prepare lots of reaction examples with known yields.  
- Convert reactions to sequences of numeric tokens (characters or chemical tokens).  
- Initialize an RNN model (random weights).  
- Repeatedly do forward passes to get predictions, compute loss (e.g., MSE between predicted and true yield), backpropagate errors, and update weights.  
- Over many iterations, the RNN parameters adjust to minimize the error on training reactions.  
  
The result is a trained RNN model that, when given a new reaction (that it hasn’t seen), can process the sequence and output a predicted yield. Importantly, the model’s hidden state approach means it has learned which parts of the sequence matter – effectively which functional groups or sequence patterns correlate with high or low yield, even though we didn’t explicitly hard-code any chemistry rules.

### Advantages
- **Captures Temporal Patterns**: RNNs excel at modeling sequential data. They can learn how earlier time points in a reaction influence later time points and final results.  
- **Dynamic Prediction**: As the RNN processes each new measurement (e.g. hourly temperature or conversion), you can “pause” the sequence and ask it for a yield estimate based on what it’s seen so far. For instance, after 2 hours of data the RNN might predict a 40 % final yield; by hour 4 it updates that to 55 %; and at hour 8 it refines to 65 %. This lets you make in fast decisions like add more catalyst or change temperature before the reaction finishes.  
- **Flexible Input Length**: RNNs can handle sequences of varying length by processing one step at a time, making them versatile for different processes.  

### Limitations
- **Data Requirements**: Training RNNs typically requires a substantial amount of sequential data. If most reactions in your dataset are only recorded as final yields (no time series of intermediate data), RNNs can’t be directly applied.  
- **Vanishing Gradients on Long Sequences**: Standard RNNs can struggle with very long sequences due to vanishing or exploding gradients, making it hard to learn long-range dependencies. Architectural improvements like LSTM mitigate this but add complexity.  
- **Complexity for Beginners**: The mathematical formulation of RNNs (with looping states and backpropagation through time) is more complex than a simple feed-forward network. This can be challenge for those new to machine learning.  

### Implementation Example: RNN for Yield Prediction

Let’s walk through an example of building and training an RNN model to predict reaction yields using
Python and PyTorch. We will use a real-world dataset for demonstration. For instance, the Buchwald–
Hartwig amination HTE dataset (Ahneman et al., Science 2018) contains thousands of C–N cross-coupling
reactions with measured yields. Each data point in this dataset is a reaction (specific aryl halide, amine,
base, ligand, etc.) and the resulting yield.  
  
#### 1. Set Up The Environment
- Open google-colab/Python  
- Instal core packages for RNN:  
```python
    pip install torch torchvision torchaudio  
    pip install pandas scikit-learn matplotlib requests  
```
- Install chemistry helpers package:  
```python
    pip intall rdkit
```
#### 2. Download the dataset  
Download Buchwald-Hartwig dataset then store in "yield_data.csv" file  
```python
    import requests

    # URL of the raw CSV data
    url = "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
    resp = requests.get(url)

    with open("yield_data.csv", "wb") as f:
        f.write(resp.content)

    print("Dataset downloaded and saved as yield_data.csv")
```

#### 3. Get helpers function and generate reaction SMILES
Transforms the spreadsheet style HTE dataset into standard reaction SMILES format  
```python 

__all__ = ['canonicalize_with_dict', 'generate_buchwald_hartwig_rxns']

from rdkit import Chem
from rdkit.Chem import rdChemReactions

def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]

def generate_buchwald_hartwig_rxns(df):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []
    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row['Aryl halide']), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])
    df['product'] = products
    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive

        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append(f"{reactants}>>{row['product']}")
    return rxns
```

#### 4. Preprocessing & Data Preparation
We will load the raw Excel file, generate reaction SMILES, split into train/test, build a character‑level vocabulary, and wrap everything in PyTorch format so that we could run on RNN model later.
```python
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1.1  Read the Excel file from `yield_data.xlsx`
df = pd.read_excel('yield_data.csv', sheet_name='FullCV_01')

# 1.2 Convert each row’s reagent columns into a single reaction SMILES string
df['rxn_smiles'] = generate_buchwald_hartwig_rxns(df)

# 1.3 Keep only the SMILES and the yield column (rename to 'yield')
data = df[['rxn_smiles', 'Output']].rename(columns={'Output': 'yield'})

# 1.4 Split into train/test
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 1.5 Build a char‑level vocabulary (PAD token at index 0)
chars = sorted({ch for smi in train_df.rxn_smiles for ch in smi})
chars = ['<PAD>'] + chars
vocab  = {ch: i for i, ch in enumerate(chars)}
PAD_IDX = 0

def encode(smi):
    “Turn a SMILES string into a list of integer token IDs.”
    return [vocab[ch] for ch in smi]

# 1.6 Define a Dataset that pads & scales yields 0–1
class RxnDataset(Dataset):
    def __init__(self, df):
        self.seqs   = [torch.tensor(encode(s), dtype=torch.long) for s in df.rxn_smiles]
        self.length = [len(s) for s in self.seqs]
        self.y      = torch.tensor(df['yield'].values / 100, dtype=torch.float32)


def collate(batch):
    seqs, lens, ys = zip(*batch)
    seq_pad = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    lens    = torch.tensor(lens)
    ys      = torch.tensor(ys)
    return seq_pad, lens, ys

train_loader = DataLoader(RxnDataset(train_df), batch_size=32, shuffle=True, collate_fn=collate)
test_loader  = DataLoader(RxnDataset(test_df),  batch_size=32, shuffle=False, collate_fn=collate)
```

#### 5. Create RNN model
We define an LSTM‑based RNN that:

- Embeds each token ID → vector  

- Runs a packed LSTM over the sequence (skipping PADs)  

- Reads out the final hidden state → linear layer → predicted yield (0–1)  


```python
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class YieldRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim,
                                  padding_idx=PAD_IDX)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim,
                             batch_first=True)
        self.fc    = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x: (batch, seq_len); lengths: (batch,)
        emb = self.embed(x) 
        packed = pack_padded_sequence(
            emb, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        # h[-1]: final hidden state of last LSTM layer → (batch, hidden_dim)
        return self.fc(h[-1]).squeeze(1)
```

#### 6. Training & Evaluation  
Here we train for 60 epochs with MSE loss (on scaled 0–1 yields), print train and validation MSE every 10 epochs.
```python
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

# 1. Set up steps
# Using GPU instead of cpu if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YieldRNN(len(vocab)).to(device)
criterion = nn.MSELoss()
optim     = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------- 3. Training loop -----------------------------------------------------
for epoch in range(1, 301):          # 60 epochs
    model.train()
    train_mse, n_train = 0.0, 0
    for X, L, y in train_loader:
        X, L, y = X.to(device), L.to(device), y.to(device)
        optim.zero_grad()
        pred = model(X, L)
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        train_mse += loss.item() * y.size(0)
        n_train   += y.size(0)
    train_mse /= n_train
    # ---- validation every 10 epochs ----
    if epoch % 10 == 0:
        model.eval(); mse = 0; n = 0
        with torch.no_grad():
            for X, L, y in test_loader:
                X, L, y = X.to(device), L.to(device), y.to(device)
                mse += criterion(model(X, L), y).item() * len(y)
                n  += len(y)
        print(f"Epoch {epoch:3d} Train MSE: {train_mse}  Val MSE: {mse/n:.4f} ")

```
#### 6. Model Evaluation on the Test Set

Once training is complete, we switch the model into evaluation mode and measure how well it predicts unseen reactions:   
```python
model.eval()
preds, trues = [], []
with torch.no_grad():
    for X, L, y in test_loader:
        X, L = X.to(device), L.to(device)
        preds.extend(model(X, L).cpu().numpy() * 100)
        trues.extend(y.numpy() * 100)

import numpy as np, matplotlib.pyplot as plt, math
preds, trues = np.array(preds), np.array(trues)
rmse = math.sqrt(((preds - trues)**2).mean())
print(f"RMSE on test set: {rmse:.1f}% yield")

plt.scatter(trues, preds, alpha=0.6, edgecolors='k')
plt.plot([0,100],[0,100],'--',c='gray'); plt.xlabel('Actual %'); plt.ylabel('Predicted %')
plt.title('Yield prediction (test set)')
plt.show()

from sklearn.metrics import r2_score

r2 = r2_score(np.array(trues), np.array(preds))
print(f"R² on test set: {r2:.3f}")
```
**Output**: After training for 300 epochs, the model converged to a stable performance on the held‑out test set:
- Validation convergence:  
    - Final validation MSE ≈ 0.0030  
    - Corresponding RMSE ≈ √0.0030 ≈ 0.055 → 5.5 % yield error  
- Test‐set metrics  
    - RMSE on test set: 5.2% yield    
    - R² on test set: 0.932  
- Interpretation  
    - An RMSE of 5.2 % means that, on average, the model’s predicted yields deviate from the true yields by just over ±5 percentage points. The level of accuracy on par with many published HTE yield‐prediction models.  
    - An R² of 0.93 indicates the network captures 93 % of the variance in experimental yields, demonstrating strong predictive power.
- Scatter plot  
![Plot](../../resource/img/yield_prediction/plot_RNN.png)

---


### Section 8.1 – Quiz Questions

#### 1) Factual Questions

##### Question 1  

Which PyTorch class provides a fully connected layer that applies a linear transformation to input data?  
**A.** `nn.Linear`   
**B.** `nn.RNN`   
**C.** `nn.Conv1d`   
**D.** `nn.BatchNorm1d`   

<details><summary>▶ Click to show answer</summary>Correct Answer: A</details>  
<details><summary>▶ Click to show explanation</summary>`nn.Linear` applies a linear transformation y = xA^T + b to the incoming data, making it the standard fully connected layer in neural networks</details>

---


##### Question 2

A common training problem with vanilla RNNs on very long sequences is:  
**A.** Over-smoothing of graph topology  
**B.** Vanishing or exploding gradients  
**C.** Excessive GPU memory during inference  
**D.** Mandatory one-hot encoding of inputs  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Gradients can shrink (or blow up) across many time-steps, making learning unstable.</details>

---

#### 2) Comprehension / Application Questions

##### Question 3

Which of the following is an example application of RNNs?  
**A.** Modeling atom-level interactions in molecular graphs  
**B.** Modeling reaction kinetics in continuous pharmaceutical manufacturing  
**C.** Classifying reaction product color from images  
**D.** Embedding molecules using graph convolutional layers  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Gated units (GRU/LSTM) mitigate vanishing gradients on long sequences.</details>

---

##### Question 4

If you change `hidden_dim` from 64 → 32 without altering anything else, you primarily reduce:  
**A.** The number of time-steps processed  
**B.** Model capacity (fewer parameters)  
**C.** Training epochs required  
**D.** Sequence length  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Smaller hidden vectors mean fewer weights and less representational power.</details>

---

## 8.2 Graph Neural Networks (GNNs)
### Section 8.2 – Quiz Questions (Graph Neural Networks)

#### 1) Factual Questions

##### Question 1  

In a molecular graph, nodes typically represent:  
**A.** Bonds  
**B.** Atoms  
**C.** Ring systems  
**D.** IR peaks  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Atoms are mapped to nodes; bonds form edges connecting them.</details>

---

##### Question 2

A key advantage of GNNs for chemistry is their ability to:  
**A.** Ignore local atomic environments  
**B.** Map variable-size molecules to fixed-length vectors  
**C.** Require explicit reaction conditions  
**D.** Operate only on tabular data  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Message passing aggregates information into a size-invariant embedding.</details>

---

#### 2) Comprehension / Application Questions

##### Question 3 

You have reaction SMILES without 3-D coordinates.  
Can you still train a GNN?  
**A.** Yes – connectivity alone often works  
**B.** No – 3-D is mandatory  
**C.** Only after DFT optimisation  
**D.** Only with image data  

<details><summary>▶ Click to show answer</summary>Correct Answer: A</details>  
<details><summary>▶ Click to show explanation</summary>Most GNNs operate on 2-D graphs derived directly from SMILES.</details>

---


## 8.3 Random Forests

### Section 8.3 – Quiz Questions (Random Forests)

#### 1) Factual Questions

##### Question 1  

Random Forests are an ensemble of:  
**A.** Linear regressors  
**B.** Decision trees  
**C.** k-Means clusters  
**D.** Support-vector machines  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>
<details><summary>▶ Click to show explanation</summary>They combine many decision trees to improve predictive accuracy and reduce overfitting compared to a single tree.</details>

---

##### Question 2  

The attribute `oob_score_` printed in the snippet reports. What does oob stand for? :  
**A.** Over-optimised benchmark  
**B.** Out-of-bag R² estimate  
**C.** Observed-only bias  
**D.** Objective batching ratio  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>
<details><summary>▶ Click to show explanation</summary>`oob_score_` uses the samples not included in each bootstrap (“out-of-bag” data) to compute an R² score, giving a built-in estimate of generalization without a separate validation set.</details>

---

##### Question 3  

Which hyper-parameter chiefly controls tree diversity in a Random Forest?  
**A.** `n_estimators`  
**B.** `criterion`  
**C.** `max_depth`  
**D.** `random_state`  

<details><summary>▶ Click to show answer</summary>Correct Answer: A</details>  
<details><summary>▶ Click to show explanation</summary>More estimators = more trees, boosting ensemble variance reduction.</details>

---

#### 2) Comprehension / Application Questions

##### Question 4  

Why are Random Forests a popular baseline for small tabular datasets?  
**A.** They need deep chemical insight  
**B.** They train quickly with minimal tuning  
**C.** They require sequential temperature data  
**D.** They embed quantum mechanics  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>
<details><summary>▶ Click to show explanation</summary>Random Forests handle varied feature types, are robust to noise, and generally perform well out of the box with little hyperparameter tuning.</details>

---

##### Question 5  

Your forest overfits. Which tweak most likely reduces overfitting?  
**A.** Increase `max_depth`  
**B.** Decrease `max_depth`  
**C.** Disable bootstrapping  
**D.** Remove feature engineering  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Shallow trees generalise better by limiting each tree’s complexity.</details>

---

## 8.4 Neural Networks

### Section 8.4 – Quiz Questions (Feed-forward Neural Networks)

#### 1) Factual Questions

##### Question 1  

Which activation function is explicitly used in the MLP snippet?  
**A.** Sigmoid  
**B.** Tanh  
**C.** ReLU  
**D.** Softmax  

<details><summary>▶ Click to show answer</summary>Correct Answer: C</details>
<details><summary>▶ Click to show explanation</summary>The code uses `nn.ReLU()` between layers to introduce non-linearities.</details>

---

##### Question 2  

The MLP architecture shown contains how many hidden layers?  
**A.** 1  
**B.** 2  
**C.** 3  
**D.** 0  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Two `nn.Linear` layers sit between input and output.</details>

---

##### Question 3  

A key limitation of generic feed-forward NNs in chemistry is:  
**A.** Inability to model non-linear relations  
**B.** Need for large datasets to avoid overfitting  
**C.** Zero computational cost  
**D.** Mandatory graph inputs  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>
<details><summary>▶ Click to show explanation</summary>Neural networks have many parameters and can easily overfit small datasets, requiring large amounts of data for reliable generalization.</details>

---

#### 2) Comprehension / Application Questions

##### Question 4  

Doubling every hidden-layer size without adding data mainly risks:  
**A.** Vanishing gradients  
**B.** Underfitting  
**C.** Overfitting  
**D.** Slower I/O  

<details><summary>▶ Click to show answer</summary>Correct Answer: C</details>
<details><summary>▶ Click to show explanation</summary>Increasing model capacity without more data often leads to overfitting, where the network memorizes noise instead of learning general patterns.</details>

---

##### Question 5  

What operation does `nn.MSELoss()` perform behind the scenes?  
**A.** Computes cross-entropy between predicted probabilities and one-hot targets  
**B.** Calculates the average of squared differences between predicted and actual values  
**C.** Computes binary cross-entropy loss for binary classification  
**D.** Calculates negative log-likelihood of predicted distributions  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>MSELoss measures how far off your predictions are by taking each prediction’s error (prediction minus true value), squaring it (to penalize larger mistakes more), and then averaging all those squared errors into one overall score.</details>