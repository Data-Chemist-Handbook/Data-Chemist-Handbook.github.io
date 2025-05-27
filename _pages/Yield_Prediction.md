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

Below you’ll meet four common model families, learn why chemists use them, see a **very small PyTorch demo** for each deep‑learning model, and get a feel for their strengths and trade‑offs.


## 8.1 Recurrent Neural Networks (RNNs)

### Why RNNs make chemical sense  
* Reaction conditions (temperature ramps, reagent feeds, pH drift) are **temporal sequences**.  
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
- Difficulty in handling long sequences (long sequences can “forget” early events).  
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

## 8.2 Graph Neural Networks (GNNs)

### Why GNNs make chemical sense  
Chemists already think of molecules as **graphs** (atoms = nodes, bonds = edges).  
A GNN learns *how* local structure influences yield under a given catalytic system.

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

X = np.random.rand(100, 6)   # 6 engineered features (e.g., MW, logP, base pKa…)
y = np.random.rand(100)      # yields 0‑1

rf = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=0)
rf.fit(X, y)

print("OOB R² (quick sanity check):", rf.oob_score_)
print("sample prediction:", rf.predict(X[:1])[0])
```

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

## Section 8.5 Quiz – Yield Prediction Models  

### **Factual Questions**

1. **Which data format is most naturally consumed by a Graph Neural Network (GNN)?**  
   - A. Fixed‑length one‑hot vectors  
   - B. Time‑stamped CSV logs  
   - C. Graphs where nodes represent atoms and edges represent bonds 
   - D. Pixel grids from reaction photos  

<details><summary>Answer</summary>

**C**

</details>

<details><summary>Explanation</summary>

GNN layers (e.g., `GCNConv`) propagate information along edges of a graph, matching the way chemists draw molecules.

</details>

---

2. An RNN is especially useful when your reaction **input features vary across time** because it:  
   - A. Requires fewer floating‑point operations  
   - B. Carries previous state from one time‑step to the next  
   - C. Forces all sequences to the same length by padding  
   - D. Automatically converts °C to K  

<details><summary>Answer</summary>

**B**

</details>

<details><summary>Explanation</summary>

The recurrent connection lets the network “remember” earlier steps like a temperature ramp or feed rate.

</details>

---

3. In the PyTorch RNN demo, the tensor `x` has shape `(batch, seq_len, features)`.  
   If `seq_len` = 10, what does “10” represent here?  
   - A. Number of atoms in the molecule  
   - B. Number of recorded time‑steps per reaction  
   - C. Hidden‑layer size  
   - D. Number of epochs  

<details><summary>Answer</summary>

**B**

</details>

<details><summary>Explanation</summary>

Each reaction was logged every hour for 10 h, giving 10 sequential observations.

</details>

---

### **Comprehension / Application Questions**

4. You have **300 reactions** described by eight tabular descriptors (no time series, no structural graphs).  
   Which model is the most convenient **baseline**?  
   - A. RNN  
   - B. GNN  
   - C. Random Forest
   - D. Transformer LM  

<details><summary>Answer</summary>

**C**

</details>

<details><summary>Explanation</summary>

Random Forests train fast, require little tuning, and handle small tabular datasets well.

</details>

---

5. Your chem‑E colleague says “Yield plummets whenever the temperature ramp overshoots.”  
   Which model family is **most likely** to capture that time‑dependent failure?  
   - A. Random Forest  
   - B. RNN
   - C. GNN  
   - D. MLP with only molecular descriptors  

<details><summary>Answer</summary>

**B**

</details>

<details><summary>Explanation</summary>

RNNs specialise in sequential data and can model the effect of overshoot at particular time‑steps.

</details>

