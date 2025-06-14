---
title: 8. Yield Prediction
author: Haomin
date: 2024-08-18
category: Jekyll
layout: post
---
dataset: BH (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00579-z)

# Section 8: Yield Prediction

Yield prediction in chemistry involves estimating the outcomes of chemical reactions under specific conditions. Modern machine learning methods, including RNNs, GNNs, Random Forests, and Neural Networks, have significantly enhanced yield predictions by capturing complex reaction patterns from experimental data.

Below you’ll meet four common model families, learn why chemists use them, see a very small PyTorch demo for each deep‑learning model, and get a feel for their strengths and trade‑offs.


## 8.1 Recurrent Neural Networks (RNNs)

### Why RNNs make chemical sense  
* Reaction conditions (temperature ramps, reagent feeds, pH drift) are temporal sequences.  
* An RNN “remembers” what happened at \(t-1\) when it predicts the yield trend at \(t\).  

### Concept
- RNNs are designed to process sequential data, making them suitable for chemical processes where the reaction conditions evolve over time.  
- They consider past information (previous steps) to predict future outcomes, essential in dynamic chemical systems.

### Example Application
RNNs have been used to model chemical reaction kinetics in continuous pharmaceutical manufacturing, where stable control over complex kinetics is necessary.

### Advantages
- Good at modeling sequences (like reaction progress).  
- Captures temporal dependencies (time-dependent conditions).

### Limitations
- Difficulty in handling long sequences (long sequences can “forget” early events) or usually being called vanish gradients.  
- Complex for beginners in terms of mathematical understanding.

### Simple Python Snippet
```python
import torch, torch.nn as nn, torch.optim as optim
torch.manual_seed(0)

# toy dataset: 20 reactions, logged every hour for 10 h (1 feature each time‑step)
x = torch.randn(20, 10, 1)   # (batch, seq_len, features)
y = torch.randn(20, 1)       # final isolated yield

rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
readout = nn.Linear(32, 1)

def fwd(seq):                # last hidden state → yield
    out, _ = rnn(seq)
    return readout(out[:, -1, :])

loss_fn = nn.MSELoss()
opt = optim.Adam(list(rnn.parameters()) + list(readout.parameters()), lr=1e-2)

for epoch in range(60):
    pred = fwd(x)
    loss = loss_fn(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()

print("final training loss:", loss.item())
print("first three predicted yields:", pred[:3].flatten().tolist())
```

---

### Section 8.1 – Quiz Questions

#### 1) Factual Questions

##### Question 1  

Which PyTorch class provides a recurrent layer that processes one time-step at a time?   
**A.** `nn.Linear`   
**B.** `nn.RNN`   
**C.** `nn.Conv1d`   
**D.** `nn.BatchNorm1d`   

<details><summary>▶ Click to show answer</summary>Correct Answer: B </details>  
<details><summary>▶ Click to show explanation</summary>`nn.RNN` is the basic recurrent layer in PyTorch.</details>

---

##### Question 2  

The tensor `x` in the RNN demo has shape `(20, 10, 1)`.  
What does the middle dimension (10) correspond to?  
**A.** Batch size  
**B.** Hidden-state size  
**C.** Number of recorded time-steps per reaction  
**D.** Number of engineered features  

<details><summary>▶ Click to show answer</summary>Correct Answer: C</details>  
<details><summary>▶ Click to show explanation</summary>Each reaction was logged hourly for 10 h, giving 10 sequential observations.</details>

---

##### Question 3  

A common training problem with vanilla RNNs on very long sequences is:  
**A.** Over-smoothing of graph topology  
**B.** Vanishing or exploding gradients  
**C.** Excessive GPU memory during inference  
**D.** Mandatory one-hot encoding of inputs  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Gradients can shrink (or blow up) across many time-steps, making learning unstable.</details>

---

#### 2) Comprehension / Application Questions

##### Question 4  

Which of the following is an example application of RNNs?  
**A.** Modeling atom-level interactions in molecular graphs  
**B.** Modeling reaction kinetics in continuous pharmaceutical manufacturing  
**C.** Classifying reaction product color from images  
**D.** Embedding molecules using graph convolutional layers  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Gated units (GRU/LSTM) mitigate vanishing gradients on long sequences.</details>

---

##### Question 5  

If you change `hidden_size` from 32 → 8 without altering anything else, you primarily reduce:  
**A.** The number of time-steps processed  
**B.** Model capacity (fewer parameters)  
**C.** Training epochs required  
**D.** Sequence length  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Smaller hidden vectors mean fewer weights and less representational power.</details>

---

## 8.2 Graph Neural Networks (GNNs)

### Why GNNs make chemical sense  
Chemists already think of molecules as graphs (atoms = nodes, bonds = edges).  
A GNN learns how local structure influences yield under a given catalytic system.

### Concept
- GNNs handle data represented as graphs, perfect for chemical structures (atoms as nodes, bonds as edges).  
- They process molecular structures directly, learning from molecular connectivity.

### Example Application
Predicting chemical reaction yields by learning directly from molecular graphs to understand how specific chemical structures influence outcomes.

### Advantages
- Directly applicable to chemical structures.  
- Strong at capturing relationships and interactions in molecules.

### Limitations
- Require substantial amounts of structured molecular data.  
- Mathematical complexity might be high for beginners.

### Simple Python Snippet
```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class YieldGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = GCNConv(10, 16)
        self.h2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_idx = data.x, data.edge_index
        x = F.relu(self.h1(x, edge_idx))
        return self.h2(x, edge_idx).mean(dim=0)  # graph-level output

# tiny dummy molecule: 4 atoms, random 10‑dim features
x = torch.rand((4, 10))
edge_index = torch.tensor([[0, 1, 2, 3, 0, 2],
                           [1, 0, 3, 2, 2, 0]], dtype=torch.long)
sample_mol = Data(x=x, edge_index=edge_index)
model = YieldGNN()

pred_yield = model(sample_mol)
print("predicted yield for toy molecule:", pred_yield.item())
```

---

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

The edge_index tensor in PyTorch Geometric lists:  
**A.** Node labels and hybridisations  
**B.** Start- and end-atom indices for each bond  
**C.** Atomic numbers only  
**D.** 3-D coordinates  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>`edge_index` encodes graph connectivity as pairs of node indices.</details>

---

##### Question 3  

A key advantage of GNNs for chemistry is their ability to:  
**A.** Ignore local atomic environments  
**B.** Map variable-size molecules to fixed-length vectors  
**C.** Require explicit reaction conditions  
**D.** Operate only on tabular data  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Message passing aggregates information into a size-invariant embedding.</details>

---

#### 2) Comprehension / Application Questions

##### Question 4  

You have reaction SMILES without 3-D coordinates.  
Can you still train a GNN?  
**A.** Yes – connectivity alone often works  
**B.** No – 3-D is mandatory  
**C.** Only after DFT optimisation  
**D.** Only with image data  

<details><summary>▶ Click to show answer</summary>Correct Answer: A</details>  
<details><summary>▶ Click to show explanation</summary>Most GNNs operate on 2-D graphs derived directly from SMILES.</details>

---

##### Question 5  

If each node feature vector has length 10 and there are 4 atoms, the shape of data.x is:  
**A.** `(10, 4)`  
**B.** `(4, 10)`  
**C.** `(4,)`  
**D.** `(10,)`  

<details><summary>▶ Click to show answer</summary>Correct Answer: B</details>  
<details><summary>▶ Click to show explanation</summary>Format is `[num_nodes, num_features]` → 4 × 10.</details>

---

## 8.3 Random Forests

### Concept
Random Forests are ensemble methods combining multiple decision trees. They aggregate predictions from various decision trees to enhance accuracy and reduce overfitting.

### Example Application
Successfully predicted yields for chemical synthesis reactions such as pyrroles and dipyrromethanes, highlighting how different reagents affect reaction outcomes.

### Advantages
- Intuitive and easy to implement.  
- Good accuracy and robustness with noisy data.

### Limitations
- Less effective with extremely large datasets.  
- Interpretability can decrease with increasing complexity.

### Simple Python Snippet
```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np
np.random.seed(0)

X = np.random.rand(100, 6)   # 6 engineered features (e.g: MW, logP, base pKa…)
y = np.random.rand(100)      # yields 0‑1

rf = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=0)
rf.fit(X, y)

print("OOB R² :", rf.oob_score_)
print("sample prediction:", rf.predict(X[:1])[0])
```

---

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

### Concept
Neural networks mimic biological neurons and connections, capable of capturing nonlinear and complex relationships. They typically consist of input layers, hidden layers, and output layers.

### Example Application
Used broadly to predict chemical reaction outcomes, learning from experimental data to forecast yields even for reactions with complex mechanisms.

### Advantages
- Powerful in modeling complex, nonlinear relationships.  
- Flexible architectures applicable to various chemical prediction tasks.

### Limitations
- Requires significant computational resources.  
- Needs large datasets for effective training.

### Simple Python Snippet
```python
import torch, torch.nn as nn, torch.optim as optim
torch.manual_seed(1)

X = torch.rand(120, 8)   # 8 engineered descriptors per reaction
y = torch.rand(120, 1)   # yields

model = nn.Sequential(
    nn.Linear(8, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1)
)

loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(80):
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    opt.zero_grad(); loss.backward(); opt.step()

print("final loss:", loss.item())
print("predicted yield (first sample):", y_hat[0].item())
```

---

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








