---
title: 8. Yield Prediction
author: Haomin-QuangDao
date: 2024-08-18
category: Jekyll
layout: post
---

**Chemical Reaction Yield**: In any chemical reaction, the yield refers to the fraction (often expressed as a percentage) of reactants successfully converted to the desired product. Predicting reaction yields is crucial for chemists – a high predicted yield can save time and resources by guiding which experiments to pursue, while a low predicted yield might signal an inefficient route. Traditionally, chemists have used domain knowledge, intuition, or trial-and-error to estimate yields. However, modern machine learning methods can learn patterns from data and make fast quantitative yield predictions. In this section, we explore how several machine learning models can be applied to reaction yield prediction.  

**Machine Learning Mode**:Predicting reaction yield can be tackled with a range of models, from classical ensemble methods to deep neural networks. Here we focus on four types, Recurrent Neural Networks (RNNs), Graph Neural Networks (GNNs), Random Forests, and Feed-Forward Neural Networks, and discuss why each is suited to yield prediction in chemistry. For each model type, we explain its role in yield prediction, provide chemistry-focused examples, include simple Python code demonstrations (using the Buchwald-Hartwig dataset to evaluate the accuracy of model), and compare typical performance on benchmark datasets. By the end, you’ll see how these models transform chemical information (like molecules or reaction conditions) into a yield prediction, and understand the pros and cons of each approach.  

---

## 8.1 Recurrent Neural Networks (RNNs)  

### Why using RNN for Yield Prediction?    
One way to represent a chemical reaction is as a sequence of characters or tokens. For example, a reaction SMILES string that lists reactants, reagents, and products. A Recurrent Neural Network (RNN) is naturally suited to sequential data. It processes input one token at a time and maintains an internal memory of what came before. This makes RNNs powerful for capturing the context in a reaction sequence that might influence yield.  

In yield prediction, treating a reaction like a “sentence” allows the model to learn subtle order-dependent patterns. For instance, an RNN reading a reaction SMILES might learn that seeing an aryl bromide token early on (e.g. Br on a benzene ring) often correlates with higher yields, whereas encountering a bulky substituent token later might signal steric hindrance and thus lower yield. Traditional models that use reactions as unordered sets of features could miss such sequence context relationships.  

RNNs were among the early deep learning models applied to reaction informatics. They brought a new ability: to learn directly from raw text representations of reactions rather than relying on hand-crafted descriptors. This sequential approach can know how the arrangement of reactant fragments and conditions relate to outcomes like yield.  

### How RNNs Work?  
An RNN can be imagined as a reader moving through the reaction, updating its memory at each step. At any point in the sequence, the RNN has a hidden state summarizing all the tokens seen so far. When a new token is read, the hidden state is updated. In our context, the tokens could be atomic symbols, bonds, or even entire molecular fragments in a tokenized SMILES. The hidden state after reading the full sequence becomes a learned representation of the entire reaction, which the RNN can then map to a yield prediction.  

For example, consider an RNN analyzing the reaction “aryl bromide + amine → coupling product” under certain conditions. Initially, the RNN’s hidden state is empty. It reads “Br” (bromide) and updates its state to reflect a very reactive leaving group is present (often a sign of a favorable reaction). Next it reads the amine component; if the amine is hindered, the RNN adjusts its state to account for possible lower reactivity. As it reads the catalyst and base tokens, it further modulates its expectations, perhaps recognizing a particularly effective ligand or an incompatible additive. By the end of the sequence, the RNN’s hidden vector encodes a combination of all these factors. This vector is then passed to an output layer to produce a number: the predicted yield.    

#### Key characteristics of RNNs  
- **Sequence Awareness**: RNNs process input sequences in order, which is useful in yield prediction because the order of components (and the context they appear in) can matter. For example, an RNN can learn that a brominated reactan at the start of a SMILES might indicate a different reactivity pattern than chlorinated due to bromine being a better leaving group.  
- **Hidden State (Memory)**: The hidden state acts like a chemist’s running notebook, remembering earlier parts of the reaction. This helps capture long-range effects (e.g., something at the start of a SMILES string affecting the outcome at the end). Advanced variants like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) have special gating mechanisms to better preserve important information over long sequences. These have been used in chemistry to ensure that crucial early tokens (e.g., a particular functional group or catalyst identity) are not forgotten by the time the network reaches the yield output.  

![RNN Diagram](../../resource/img/yield_prediction/RNN_Process.png)
Figure 1: The reaction is encoded as a sequence of tokens. At each step t, the input token 𝑥_𝑡 (e.g., “Br”, “Ph”, “R”) enters an RNN cell. That cell combines the new input with the previous hidden state hₜ₋₁ to produce the updated hidden state hₜ. Arrows show the flow: horizontal arrows carry the hidden state forward through time, and vertical arrows inject each new chemistry token into the cell. This illustrates how the network incrementally builds a memory of the reaction’s components, accumulating context that will ultimately be used to predict the reaction’s yield.  

In summary, an RNN reads a reaction like a chemist reads a procedure, forming an expectation of yield as it goes. This approach can capture nuanced patterns (like synergistic effects of reagents) that might be hard to encode with fixed descriptors.  

### Why RNNs Succeed (and Where They Struggle)  
#### Advantage    
RNNs excel at capturing sequential patterns. In chemistry, this means they naturally take into account the presence and context of each component in a reaction. They can, for example, learn that “having ligand X together with base Y” is a pattern associated with high yield, whereas models that treat X and Y as independent features might miss that interaction.    

#### Limitations  
- **Data Requirements**: RNNs are data-hungry and require many examples to train well. If we had a much smaller dataset, an RNN might overfit or fail to learn meaningful patterns.    
- **Vanishing Gradients on Long Sequences**: Standard RNNs can struggle with very long sequences due to vanishing or exploding gradients, making it hard to learn long-range dependencies. Architectural improvements like LSTM mitigate this but add complexity.  
- **Complexity for Beginners**: The mathematical formulation of RNNs (with looping states and backpropagation through time) is more complex than a simple feed-forward network. This can be challenge for those new to machine learning.  

### Implementation Example: RNN for Yield Prediction

Let’s walk through an example of building and training an RNN model to predict reaction yields using Python and PyTorch. We will use a real-world dataset for demonstration. For instance, the Buchwald– Hartwig amination HTE dataset (Ahneman et al., Science 2018) contains thousands of C–N cross-coupling reactions with measured yields. Each data point in this dataset is a reaction (specific aryl halide, amine, base, ligand, etc.) and the resulting yield.  
  
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
    url = "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
    resp = requests.get(url)
    with open("yield_data.csv", "wb") as f:
        f.write(resp.content)
    print("Dataset downloaded and saved as yield_data.csv")
```  

#### 3. Get helpers function and generate reaction SMILES
Transforms the spreadsheet style HTE dataset into standard reaction SMILES format  
```python 
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
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self,idx):
        return self.seqs[idx], self.length[idx], self.y[idx]


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
Here we train for 300 epochs with MSE loss (on scaled 0–1 yields), print train and validation MSE every 10 epochs.
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
