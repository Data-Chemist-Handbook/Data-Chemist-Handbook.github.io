---
title: 3-C. Machine Learning Models
author: Haomin
date: 2024-08-13
category: Jekyll
layout: post
---

## 3.3 Graph Neural Networks

#### Completed and Compiled Code (3.3.1–3.3.3): [Click Here](https://colab.research.google.com/drive/1H3Qskx43x5t7iz7LqUSC-uLfIkwGAEET?usp=sharing)

A **Graph Neural Network (GNN)** is built for **graph-structured data**. Nodes carry features, edges carry relationships, and learning proceeds by **passing messages** along edges, **updating node states**, and then **reading out** a graph-level vector for a downstream task.

**Structure.** Layers alternate between (i) **message passing**—neighbors send transformed signals along edges—and (ii) **state updates**—each node revises its embedding with a nonlinear function. Stacking (k) layers exposes a node to its (k)-hop neighborhood. A final **readout/pooling** (e.g., sum/mean) aggregates node embeddings into a fixed-length molecular representation.

**Functioning.** In molecular graphs, **atoms = nodes** and **bonds = edges**. Messages combine atom features with bond features, propagate to destination atoms, and are transformed (e.g., by MLPs or GRUs). After several layers, each atom’s embedding reflects its **local chemical context** (hybridization, aromaticity, ring membership, nearby heteroatoms).

**Learning.** Training minimizes a task loss (e.g., BCEWithLogits for classification) with gradient descent. Gradients flow **through the graph topology**, tuning how atoms attend to their neighbors and how the pooled representation supports the final prediction.

**Roadmap.**

3.3.1 looks at **OGB-MOLHIV**, the dataset and official split, and builds loaders with basic EDA.

3.3.2 implements an **edge-aware MPNN** (NNConv + GRU) and reads the curves/ROC.

3.3.3 compares **GCN** and **MPNN** on the same split and discusses the outcomes.

---

### 3.3.1 From Descriptors to Molecular Graphs: OGB-MOLHIV at a Glance

**Why start here.** Descriptor-only QSAR can flatten connectivity; a molecular GNN keeps the **who-is-bonded-to-whom** information. The **OGB-MOLHIV** benchmark (Hu et al., 2020) provides graph data (atoms, bonds) with an **anti-HIV activity** label and an **official train/valid/test split**—ideal for a clean, reproducible pipeline.

<div style="background-color:#e3f2fd; padding:15px; border-radius:8px; margin:15px 0;">
  <table style="width:100%; border-collapse:collapse;">
    <tr style="background-color:#90caf9;">
      <th style="padding:10px; border:1px solid #42a5f5; text-align:center;" colspan="3">Key Functions and Concepts (Data Layer)</th>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>PygGraphPropPredDataset</b><br>
        OGB–PyG dataset wrapper<br>
        <span style="font-size:0.9em; color:#666;">Auto-download + official split</span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>DataLoader (PyG)</b><br>
        Mini-batches of graphs<br>
        <span style="font-size:0.9em; color:#666;">Collates <i>Data</i> objects into <i>DataBatch</i></span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>DataBatch fields</b><br>
        <code>x, edge_index, edge_attr, y</code><br>
        <span style="font-size:0.9em; color:#666;">Atoms (9-dim), bonds (3-dim), labels</span>
      </td>
    </tr>
  </table>
</div>

#### 3.3.1-A  Load the dataset and the official split

*What this cell does.* Imports minimal packages, downloads/loads `ogbg-molhiv`, and fetches the official indices for train/valid/test.

```python
# Minimal imports
import numpy as np, matplotlib.pyplot as plt, torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

# Load + split
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
split = dataset.get_idx_split()
train_set, valid_set, test_set = dataset[split["train"]], dataset[split["valid"]], dataset[split["test"]]

print(dataset)
print(f"Graphs: total={len(dataset)} | train/valid/test = {len(train_set)}/{len(valid_set)}/{len(test_set)}")
```

> **Results**
> `PygGraphPropPredDataset(41127)`
> `Graphs: total=41127 | train/valid/test = 32901/4113/4113`

This confirms the canonical OGB split we use throughout.

#### 3.3.1-B  Quick EDA (label skew, nodes/edges per graph)

*What this cell does.* Builds three histograms: label distribution, atoms per molecule, and bonds per molecule—so we know the class balance and a reasonable message-passing depth.

```python
labels = dataset.data.y.view(-1).cpu().numpy()
num_nodes = [g.num_nodes for g in dataset]
num_edges = [g.num_edges for g in dataset]

fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), dpi=140)
axs[0].hist(labels, bins=[-0.5,0.5,1.5], rwidth=0.8, edgecolor="black"); axs[0].set_xticks([0,1])
axs[0].set_title("Label distribution"); axs[0].grid(alpha=0.3)
axs[1].hist(num_nodes, bins=40, edgecolor="black"); axs[1].set_title("Nodes per graph"); axs[1].grid(alpha=0.3)
axs[2].hist(num_edges, bins=40, edgecolor="black"); axs[2].set_title("Edges per graph"); axs[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

**How to read it.** Labels are **heavily imbalanced** (positives ≪ negatives), so **ROC-AUC** is a better main metric than accuracy, and class-weighted BCE is advisable. Graph sizes are mostly in the **tens of atoms/bonds**, so **2–4 message-passing layers** are a reasonable starting point.

![molhiv\_eda](../../../../../resource/img/GNNfig/molhiv_eda.png)

#### 3.3.1-C  DataLoaders + a batch sanity check

*What this cell does.* Creates loaders and prints one training batch to confirm feature shapes (we will reuse the loaders later).

```python
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)

batch = next(iter(train_loader))
print(batch)
print("num_graphs:", batch.num_graphs,
      "| node_feat_dim:", batch.x.size(-1),
      "| edge_feat_dim:", batch.edge_attr.size(-1),
      "| y shape:", tuple(batch.y.view(-1).shape))
```

> **Results**
> `DataBatch(edge_index=[2, 3640], edge_attr=[3640, 3], x=[1711, 9], y=[64, 1], …)`
> `num_graphs: 64 | node_feat_dim: 9 | edge_feat_dim: 3 | y shape: (64,)`

This confirms the exact input dimensions we connect to the models in §3.3.2 and §3.3.3.

---

### 3.3.2 Message Passing as Chemical Reasoning (NNConv–GRU MPNN)

The **Message Passing Neural Network (MPNN)** (Gilmer et al., 2017) is a family of GNNs designed to learn directly from molecular graphs.
Each atom updates its representation by **aggregating messages** from its neighboring atoms through bonds.
In our implementation, bond features determine how messages are weighted, and a **Gated Recurrent Unit (GRU)** stabilizes multi-step updates.

**Goal.** Build and train an **edge-aware MPNN** (using `NNConv` + `GRU`) for molecular property prediction on `ogbg-molhiv`, record training dynamics, and evaluate the ROC-AUC on the held-out test set.

<div style="background-color:#e3f2fd; padding:15px; border-radius:8px; margin:15px 0;">
  <table style="width:100%; border-collapse:collapse;">
    <tr style="background-color:#90caf9;">
      <th style="padding:10px; border:1px solid #42a5f5; text-align:center;" colspan="4">Key Functions and Concepts (MPNN)</th>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>NNConv</b><br>
        Edge-conditioned convolution<br>
        <span style="font-size:0.9em; color:#666;">Uses bond features to parameterize message filters</span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>EdgeNet</b><br>
        MLP for edge feature transformation<br>
        <span style="font-size:0.9em; color:#666;">Maps edge attributes → filter matrices</span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>GRU</b><br>
        Gated recurrent unit<br>
        <span style="font-size:0.9em; color:#666;">Controls information flow across layers</span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>global_add_pool</b><br>
        Readout layer<br>
        <span style="font-size:0.9em; color:#666;">Aggregates atomic embeddings → molecule representation</span>
      </td>
    </tr>
  </table>
</div>

#### 3.3.2-A  Reproducibility, device, and class weight

This first cell ensures consistent runs and builds loaders for the model.
Because the **MOLHIV dataset is imbalanced**, we compute a positive-class weight for use in `BCEWithLogitsLoss`.

```python
import random, numpy as np, torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

# Fix randomness for reproducibility
def set_seed(s=1):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(1)

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and official splits
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
split = dataset.get_idx_split()
train_loader = DataLoader(dataset[split["train"]], batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset[split["valid"]], batch_size=256, shuffle=False)
test_loader  = DataLoader(dataset[split["test"]],  batch_size=256, shuffle=False)

# Compute class weight (for imbalance)
train_y = dataset[split["train"]].y.view(-1).cpu().numpy()
pos_weight = torch.tensor([(train_y==0).sum()/(train_y==1).sum()], dtype=torch.float, device=device)

D_x, D_e = dataset.num_node_features, dataset.num_edge_features
```

**Explanation.**
This setup guarantees that experiments are repeatable.
`pos_weight` > 1 tells the loss function to assign higher penalty to misclassified positive samples.

#### 3.3.2-B  Define the model: EdgeNet + NNConv + GRU

This section builds the network architecture.
Each message-passing layer applies an **edge-specific linear transformation** followed by a GRU update and dropout for regularization.

```python
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import NNConv, global_add_pool

class EdgeNet(nn.Module):
    """Small MLP that transforms edge features into NNConv filters."""
    def __init__(self, edge_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_in, hidden*hidden),
            nn.ReLU(),
            nn.Linear(hidden*hidden, hidden*hidden)
        )
        self.hidden = hidden

    def forward(self, e):  # e: [E, edge_in]
        return self.net(e)  # Output: [E, hidden*hidden]


class MPNN(nn.Module):
    """Message Passing Neural Network with edge-conditioned messages."""
    def __init__(self, node_in, edge_in, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(node_in, hidden)
        self.edge_net = EdgeNet(edge_in, hidden)
        self.convs = nn.ModuleList([
            NNConv(hidden, hidden, self.edge_net, aggr='add') for _ in range(layers)
        ])
        self.gru = nn.GRU(hidden, hidden)
        self.dropout = dropout
        self.readout = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.float(), data.edge_index, data.edge_attr, data.batch
        # Safety guard: ensure bond features are floats
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), D_e), dtype=torch.float, device=x.device)
        else:
            edge_attr = edge_attr.float()

        # 1) Embed atom features
        x = self.embed(x)
        h = x.unsqueeze(0)

        # 2) Perform message passing and gated update
        for conv in self.convs:
            m = F.relu(conv(x, edge_index, edge_attr))
            m = F.dropout(m, p=self.dropout, training=self.training).unsqueeze(0)
            out, h = self.gru(m, h)
            x = out.squeeze(0)

        # 3) Aggregate to graph-level representation
        g = global_add_pool(x, batch)

        # 4) Linear readout → logit
        return self.readout(g).view(-1)
```

**Explanation.**

* `EdgeNet` maps 3-dimensional bond descriptors to layer-specific filters.
* `NNConv` uses those filters to compute neighbor messages.
* `GRU` decides how much new information to incorporate versus retain from previous states.
* `global_add_pool` sums over all atoms to yield a molecule-level vector.

#### 3.3.2-C  Training, validation (AUC), and testing (ROC)

The training loop minimizes BCEWithLogitsLoss, monitors **AUC on the validation split**, and restores the best weights before evaluating on the test set.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt, numpy as np

def train_epoch(model, loader, opt, crit, clip=2.0):
    """Single epoch of training."""
    model.train(); total, n = 0.0, 0
    for data in loader:
        data = data.to(device)
        y = data.y.view(-1).float()
        opt.zero_grad()
        logits = model(data)
        loss = crit(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * y.numel(); n += y.numel()
    return total / n


@torch.no_grad()
def eval_auc(model, loader):
    """Compute ROC-AUC on a loader."""
    model.eval(); y_true, y_prob = [], []
    for data in loader:
        data = data.to(device)
        prob = torch.sigmoid(model(data)).cpu().numpy()
        y_true.append(data.y.view(-1).cpu().numpy())
        y_prob.append(prob)
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return auc, (fpr, tpr)


# Initialize model, optimizer, and loss
model = MPNN(D_x, D_e, hidden=128, layers=3, dropout=0.2).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val, best_state = -1.0, None
train_curve, val_curve = [], []

# Train for 20 epochs
for ep in range(1, 21):
    tr = train_epoch(model, train_loader, opt, crit)
    va, _ = eval_auc(model, valid_loader)
    train_curve.append(tr); val_curve.append(va)
    if va > best_val:
        best_val = va
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    print(f"Epoch {ep:02d} | train {tr:.4f} | valid AUC {va:.4f}")

# Restore best checkpoint and test
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
test_auc, (fpr, tpr) = eval_auc(model, test_loader)
print(f"[MPNN] Best valid AUC = {best_val:.4f} | Test AUC = {test_auc:.4f}")

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
axs[0].plot(train_curve, label="Train loss")
axs[0].plot(val_curve, label="Valid AUC")
axs[0].set_title("Training progress")
axs[0].set_xlabel("Epoch")
axs[0].grid(alpha=0.3)
axs[0].legend()
axs[1].plot(fpr, tpr, lw=2, label=f"MPNN (AUC={test_auc:.3f})")
axs[1].plot([0,1],[0,1],"--",color="gray")
axs[1].set_xlabel("False Positive Rate")
axs[1].set_ylabel("True Positive Rate")
axs[1].set_title("ROC on test")
axs[1].grid(alpha=0.3)
axs[1].legend()
plt.tight_layout(); plt.show()
```

> **Training Log (excerpt)**
> `Epoch 17 … valid AUC 0.7089` → best
> `Test AUC = 0.7268`

**Interpretation.**

* Training loss decreases smoothly to ≈ 0.18.
* Validation AUC fluctuates between 0.35 – 0.70 due to class imbalance and small batch effects, peaking around 0.71.
* The best test AUC ≈ 0.73 shows that the model captures useful substructural information even in a compact configuration.
* The oscillations are typical for imbalanced molecular classification tasks; **early stopping** at the validation peak prevents overfitting.

![mpnn\_curves\_roc](../../../../../resource/img/GNNfig/mpnn_curves_roc.png)

**References (for §3.3 & §3.3.1 §3.3.2)**

* Fey, M., & Lenssen, J. E. (2019). *PyTorch Geometric*.
* Hu, W. et al. (2020). *Open Graph Benchmark*.
* Wu, Z. et al. (2018). *MoleculeNet*.
---

### 3.3.3 Same Split, Two Architectures: **GCN** vs **Edge-aware MPNN**

We now place two graph architectures side by side on the **same OGB-MOLHIV split** and with the **same evaluation metric (ROC-AUC)**:

* **GCN** (Kipf & Welling, 2017): neighborhood aggregation without an edge MLP; all bonds contribute uniformly.
* **MPNN (NNConv + GRU)**: messages are **edge-conditioned** by bond features; a GRU stabilizes multi-step updates.

The point is to **contrast how the two formulations behave under an identical training recipe**, not to assume one is inherently superior.

<div style="background-color:#e3f2fd; padding:15px; border-radius:8px; margin:15px 0;">
  <table style="width:100%; border-collapse:collapse;">
    <tr style="background-color:#90caf9;">
      <th style="padding:10px; border:1px solid #42a5f5; text-align:center;" colspan="4">Key Functions and Concepts (Comparison)</th>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>GCNConv</b><br>
        Graph convolution (no bond MLP)<br>
        <span style="font-size:0.9em; color:#666;">Uniform treatment of bonds during aggregation</span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>NNConv</b><br>
        Edge-conditioned convolution<br>
        <span style="font-size:0.9em; color:#666;">Bond features → per-edge filters</span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>GRU</b><br>
        Gated update across layers<br>
        <span style="font-size:0.9em; color:#666;">Stabilizes multi-step message passing</span>
      </td>
      <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
        <b>ROC-AUC</b><br>
        Ranking under class imbalance<br>
        <span style="font-size:0.9em; color:#666;">Threshold-free comparison</span>
      </td>
    </tr>
  </table>
</div>

> **Note.** This subsection reuses the loaders and `pos_weight` from §3.3.2 (3.3.2-A). If you run cells independently, execute §3.3.1 and §3.3.2-A first.

#### 3.3.3-A  Define the two architectures succinctly

*What this cell does.* Implements a compact **GCN** baseline and the **NNConv + GRU** MPNN; both end with `global_add_pool` and a linear head that outputs a **logit** per molecule.

```python
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, global_add_pool

class GCN_Baseline(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.drop  = dropout
        self.out   = nn.Linear(hidden, 1)
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.embed(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.drop, training=self.training)
        g = global_add_pool(x, batch)
        return self.out(g).view(-1)

class EdgeNet(nn.Module):
    def __init__(self, edge_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_in, hidden*hidden), nn.ReLU(),
            nn.Linear(hidden*hidden, hidden*hidden)
        ); self.hidden = hidden
    def forward(self, e): return self.net(e)

class MPNN_NNConv(nn.Module):
    def __init__(self, node_in, edge_in, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.embed   = nn.Linear(node_in, hidden)
        self.edge_net= EdgeNet(edge_in, hidden)
        self.convs   = nn.ModuleList([NNConv(hidden, hidden, self.edge_net, aggr='add') for _ in range(layers)])
        self.gru     = nn.GRU(hidden, hidden)
        self.drop    = dropout
        self.out     = nn.Linear(hidden, 1)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.float(), data.edge_index, data.edge_attr, data.batch
        edge_attr = edge_attr.float() if edge_attr is not None else torch.zeros((edge_index.size(1), D_e), device=x.device)
        x = self.embed(x); h = x.unsqueeze(0)
        for conv in self.convs:
            m = F.relu(conv(x, edge_index, edge_attr))
            m = F.dropout(m, p=self.drop, training=self.training).unsqueeze(0)
            out, h = self.gru(m, h); x = out.squeeze(0)
        g = global_add_pool(x, batch)
        return self.out(g).view(-1)
```

#### 3.3.3-B  Shared training/evaluation routine

*What this cell does.* Trains either model with the **same** loop and hyperparameters, selects the **best validation AUC** checkpoint, and reports **test AUC** and **ROC**.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np, torch

def train_and_select(model, tag, train_loader, valid_loader, test_loader, pos_weight=None, epochs=15, lr=1e-3):
    device = next(model.parameters()).device
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val, best = -1.0, None

    for ep in range(1, epochs+1):
        # Train one epoch
        model.train(); tot, n = 0.0, 0
        for data in train_loader:
            data = data.to(device); y = data.y.view(-1).float()
            opt.zero_grad(); loss = crit(model(data), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            tot += loss.item() * y.numel(); n += y.numel()

        # Validate
        model.eval(); y_true, y_prob = [], []
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                p = torch.sigmoid(model(data)).cpu().numpy()
                y_true.append(data.y.view(-1).cpu().numpy()); y_prob.append(p)
        y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
        val_auc = roc_auc_score(y_true, y_prob)
        print(f"[{tag}] epoch {ep:02d} | train {tot/n:.4f} | valid AUC {val_auc:.4f}")

        if val_auc > best_val:
            best_val = val_auc
            best = {k: v.cpu() for k, v in model.state_dict().items()}

    # Test with best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best.items()})
    model.eval(); y_true, y_prob = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            p = torch.sigmoid(model(data)).cpu().numpy()
            y_true.append(data.y.view(-1).cpu().numpy()); y_prob.append(p)
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    test_auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    print(f"[{tag}] BEST valid AUC {best_val:.4f} | TEST AUC {test_auc:.4f}")
    return test_auc, (fpr, tpr)
```

#### 3.3.3-C  Run both models and visualize

*What this cell does.* Instantiates **GCN** and **MPNN**, runs the shared routine for **15 epochs**, and produces ROC curves plus a bar chart of the two AUCs.

```python
import matplotlib.pyplot as plt

gcn  = GCN_Baseline(D_x).to(device)
mpnn = MPNN_NNConv(D_x, D_e).to(device)

gcn_auc, (gcn_fpr, gcn_tpr)   = train_and_select(gcn,  "GCN",  train_loader, valid_loader, test_loader, pos_weight, epochs=15)
mpn_auc, (mpn_fpr, mpn_tpr)   = train_and_select(mpnn, "MPNN", train_loader, valid_loader, test_loader, pos_weight, epochs=15)

# Plot ROC and AUC bars
fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
ax[0].plot(gcn_fpr, gcn_tpr, lw=2, label=f"GCN (AUC={gcn_auc:.3f})")
ax[0].plot(mpn_fpr, mpn_tpr, lw=2, label=f"MPNN/NNConv (AUC={mpn_auc:.3f})")
ax[0].plot([0,1],[0,1],"--",color="gray"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[0].set_title("ROC on test"); ax[0].grid(alpha=0.3); ax[0].legend()

ax[1].bar(["GCN","MPNN"], [gcn_auc, mpn_auc], edgecolor="black", alpha=0.9)
for x, v in zip(["GCN","MPNN"], [gcn_auc, mpn_auc]):
    ax[1].text(x, v+0.01, f"{v:.3f}", ha="center", va="bottom")
ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("AUC"); ax[1].set_title("AUC comparison")
ax[1].grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
```

> **Recorded results**
> **GCN**
> `[GCN] … BEST valid AUC 0.7212 | TEST AUC 0.6640`
> **MPNN (quick comparison recipe)**
> `[MPNN] … BEST valid AUC 0.5505 | TEST AUC 0.5894`

> **Figure**
> ![gcn\_mpn](../../../../../resource/img/GNNfig/gcn_mpn.png)

#### How to read these outcomes (with the logs)

* Under the **shared 15-epoch recipe**, the **GCN** run landed at **test AUC ≈ 0.664**, while the quick **MPNN** comparison run landed at **≈ 0.589**.
* In §3.3.2, a separate MPNN run—same split but with early-stopping at the sharp validation peak and 20 epochs—reached **test AUC ≈ 0.727**.
* Together, these runs show **how training choices (epoch budget, early-stopping point, class weighting, LR/scheduler)** influence the two architectures differently. The comparison here is useful to see relative behavior under the same budget; §3.3.2 shows what the edge-aware model can achieve with a slightly longer/steadier training loop.

**Optional dials to try (one at a time).**
Epochs 30–40 for MPNN; LR (5\times 10^{-4}) for MPNN; hidden 256 (layers=3); `ReduceLROnPlateau` on validation AUC; keep `weight_decay=1e-5`, dropout=0.2.

**References (for §3.3.3)**
Kipf & Welling, 2017 (GCN); Gilmer et al., 2017 (MPNN); Fey & Lenssen, 2019 (PyG); Hu et al., 2020 (OGB).

---

### 3.3.4 Challenges and Interpretability in GNNs

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1rtJ6voxMVK7KS-oZS97_7Jr1bBf7Z15T?usp=sharing)

<div style="background-color:#f0f7ff; border:2px solid #1976d2; border-radius:10px; padding:20px; margin:20px 0;">
    <h4>What We're Exploring: Fundamental Challenges in Graph Neural Networks</h4>
    
    <div style="background-color:#fff3e0; padding:15px; border-radius:8px; margin-bottom:15px;">
        <p><b>Why Study GNN Challenges?</b></p>
        <ul>
            <li><b>Over-smoothing:</b> Why deeper isn't always better - node features become indistinguishable</li>
            <li><b>Interpretability:</b> Understanding what the model learns - crucial for drug discovery</li>
            <li><b>Real Impact:</b> These challenges affect whether GNNs can be trusted in production</li>
        </ul>
        <p><b>What you'll learn:</b> The fundamental limitations of GNNs and current solutions to overcome them</p>
    </div>
    
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#e3f2fd;">
            <th style="padding:10px; border:1px solid #90caf9;">Challenge</th>
            <th style="padding:10px; border:1px solid #90caf9;">What Happens</th>
            <th style="padding:10px; border:1px solid #90caf9;">Why It Matters</th>
            <th style="padding:10px; border:1px solid #90caf9;">Solutions</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #90caf9; background-color:#ffebee;"><b>Over-smoothing</b></td>
            <td style="padding:10px; border:1px solid #90caf9;">Node features converge<br><span style="color:#666; font-size:0.9em;">All atoms look the same</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">Limits network depth<br><span style="color:#666; font-size:0.9em;">Can't capture long-range interactions</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">Residual connections<br><span style="color:#666; font-size:0.9em;">Skip connections, normalization</span></td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #90caf9; background-color:#fff9c4;"><b>Interpretability</b></td>
            <td style="padding:10px; border:1px solid #90caf9;">Black box predictions<br><span style="color:#666; font-size:0.9em;">Don't know why it predicts</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">No trust in predictions<br><span style="color:#666; font-size:0.9em;">Can't guide drug design</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">Attention visualization<br><span style="color:#666; font-size:0.9em;">Substructure explanations</span></td>
        </tr>
    </table>
</div>

While GNNs have shown remarkable success in molecular property prediction, they face several fundamental challenges that limit their practical deployment. In this section, we'll explore two critical issues: the over-smoothing phenomenon that limits network depth, and the interpretability challenge that makes it difficult to understand model predictions.

#### The Power of Depth vs. The Curse of Over-smoothing

In Graph Neural Networks (GNNs), adding more message-passing layers allows nodes (atoms) to gather information from increasingly distant parts of a graph (molecule). At first glance, it seems deeper networks should always perform better—after all, more layers mean more context. But in practice, there's a major trade-off known as **over-smoothing**.

<div style="background-color:#ffebee; padding:15px; border-radius:8px; margin:20px 0;">
    <h4>Understanding Over-smoothing</h4>
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#ffcdd2;">
            <th style="padding:8px; border:1px solid #ef9a9a;">Concept</th>
            <th style="padding:8px; border:1px solid #ef9a9a;">Simple Explanation</th>
            <th style="padding:8px; border:1px solid #ef9a9a;">Molecular Context</th>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ef9a9a;"><b>Message Passing</b></td>
            <td style="padding:8px; border:1px solid #ef9a9a;">Atoms share info with neighbors</td>
            <td style="padding:8px; border:1px solid #ef9a9a;">Like atoms "talking" through bonds</td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ef9a9a;"><b>Receptive Field</b></td>
            <td style="padding:8px; border:1px solid #ef9a9a;">How far information travels</td>
            <td style="padding:8px; border:1px solid #ef9a9a;">k layers = k-hop neighborhood</td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ef9a9a;"><b>Over-smoothing</b></td>
            <td style="padding:8px; border:1px solid #ef9a9a;">All nodes become similar</td>
            <td style="padding:8px; border:1px solid #ef9a9a;">Can't distinguish different atoms</td>
        </tr>
        <tr>
            <td style="padding:8px; border:1px solid #ef9a9a;"><b>Critical Depth</b></td>
            <td style="padding:8px; border:1px solid #ef9a9a;">~3-5 layers typically</td>
            <td style="padding:8px; border:1px solid #ef9a9a;">Beyond this, performance drops</td>
        </tr>
    </table>
</div>

**What to Demonstrate**

Before we jump into the code, here's **what it's trying to show**:

We want to measure how **similar node embeddings become** as we increase the number of GCN layers. If all node vectors become nearly identical after several layers, that means the model is **losing resolution**—different atoms can't be distinguished anymore. This is called **over-smoothing**.

<div style="background-color:#e3f2fd; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#90caf9;">
            <th style="padding:10px; border:1px solid #42a5f5; text-align:center;" colspan="4">Key Functions and Concepts</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>GCNConv</b><br>
                Graph convolution layer<br>
                <span style="font-size:0.9em; color:#666;">Aggregates neighbor features</span>
            </td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>F.relu()</b><br>
                Non-linear activation<br>
                <span style="font-size:0.9em; color:#666;">Adds expressiveness</span>
            </td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>F.normalize()</b><br>
                L2 normalization<br>
                <span style="font-size:0.9em; color:#666;">For cosine similarity</span>
            </td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>torch.mm()</b><br>
                Matrix multiplication<br>
                <span style="font-size:0.9em; color:#666;">Computes similarity matrix</span>
            </td>
        </tr>
    </table>
</div>

**Functions and Concepts Used**

* **`GCNConv` (from `torch_geometric.nn`)**: This is a standard Graph Convolutional Network (GCN) layer. It performs message passing by aggregating neighbor features and updating node embeddings. It normalizes messages by node degrees to prevent high-degree nodes from dominating.

* **`F.relu()`**: Applies a non-linear ReLU activation function after each GCN layer. This introduces non-linearity to the model, allowing it to learn more complex patterns.

* **`F.normalize(..., p=2, dim=1)`**: This normalizes node embeddings to unit length (L2 norm), which is required for cosine similarity calculation.

* **`torch.mm()`**: Matrix multiplication is used here to compute the full cosine similarity matrix between normalized node embeddings.

* **Cosine similarity**: Measures how aligned two vectors are (value close to 1 means very similar). By averaging all pairwise cosine similarities, we can track whether the node representations are collapsing into the same vector.

**Graph Construction**

We use a **6-node ring structure** as a simple molecular graph. Each node starts with a unique identity (using identity matrix `torch.eye(6)` as input features), and all nodes are connected in a cycle:

<div style="background-color:#e8f5e9; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#a5d6a7;">
            <th style="padding:10px; border:1px solid #66bb6a; text-align:center;" colspan="4">Graph Construction Process</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Step 1:</b><br>
                Create node features<br>
                <span style="font-size:0.9em; color:#666;">Identity matrix (6×6)</span>
            </td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Step 2:</b><br>
                Define ring topology<br>
                <span style="font-size:0.9em; color:#666;">Each node → 2 neighbors</span>
            </td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Step 3:</b><br>
                Make bidirectional<br>
                <span style="font-size:0.9em; color:#666;">12 directed edges total</span>
            </td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Result:</b><br>
                PyG Data object<br>
                <span style="font-size:0.9em; color:#666;">Ready for GNN</span>
            </td>
        </tr>
    </table>
</div>

```python
import torch
from torch_geometric.data import Data

# Each node has a unique 6D feature vector (identity matrix)
x = torch.eye(6)

# Define edges for a 6-node cycle (each edge is bidirectional)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 0, 5, 0, 1, 2, 3, 4]
], dtype=torch.long)

# Create PyTorch Geometric graph object
data = Data(x=x, edge_index=edge_index)
```

**Over-smoothing Analysis**

Now we apply the same GCN layer multiple times to simulate a deeper GNN. After each layer, we re-compute the node embeddings and compare them using cosine similarity:

<div style="background-color:#f3e5f5; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#e1bee7;">
            <th style="padding:10px; border:1px solid #ba68c8; text-align:center;" colspan="3">Over-smoothing Measurement Process</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ce93d8; background-color:#ffffff;">
                <b>Apply GCN layers:</b><br>
                Stack 1-10 layers<br>
                <span style="font-size:0.9em; color:#666;">Same layer repeated</span>
            </td>
            <td style="padding:10px; border:1px solid #ce93d8; background-color:#ffffff;">
                <b>Compute similarity:</b><br>
                Cosine between nodes<br>
                <span style="font-size:0.9em; color:#666;">Average all pairs</span>
            </td>
            <td style="padding:10px; border:1px solid #ce93d8; background-color:#ffffff;">
                <b>Track convergence:</b><br>
                Plot vs depth<br>
                <span style="font-size:0.9em; color:#666;">Watch similarity → 1</span>
            </td>
        </tr>
    </table>
</div>

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv

def measure_smoothing(num_layers, data):
    """
    Apply num_layers GCNConv layers and measure
    how similar node embeddings become.
    """
    x = data.x
    for _ in range(num_layers):
        conv = GCNConv(x.size(1), x.size(1))
        x = F.relu(conv(x, data.edge_index))

    # Normalize embeddings for cosine similarity
    x_norm = F.normalize(x, p=2, dim=1)
    
    # Cosine similarity matrix
    similarity_matrix = torch.mm(x_norm, x_norm.t())
    
    # Exclude diagonal (self-similarity) when averaging
    n = x.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    avg_similarity = similarity_matrix[mask].mean().item()
    
    return avg_similarity

# Run for different GNN depths
depths = [1, 3, 5, 10]
sims = []
for depth in depths:
    sim = measure_smoothing(depth, data)
    sims.append(sim)
    print(f"Depth {depth}: Average similarity = {sim:.3f}")

# Plot the smoothing effect
plt.plot(depths, sims, marker='o')
plt.xlabel("Number of GCN Layers")
plt.ylabel("Average Cosine Similarity")
plt.title("Over-smoothing Effect in GNNs")
plt.grid(True)
plt.show()
```

**Output**
```
Depth 1: Average similarity = 0.406
Depth 3: Average similarity = 0.995
Depth 5: Average similarity = 0.993
Depth 10: Average similarity = 1.000
```

![Over-smoothing in GNNs](../../resource/img/gnn/oversmoothing.png)

<div style="background-color:#fff9c4; padding:15px; border-radius:8px; margin:15px 0;">
    <h4>Interpretation of Results</h4>
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#fff176;">
            <th style="padding:10px; border:1px solid #ffd600;">Depth</th>
            <th style="padding:10px; border:1px solid #ffd600;">Similarity</th>
            <th style="padding:10px; border:1px solid #ffd600;">What It Means</th>
            <th style="padding:10px; border:1px solid #ffd600;">Practical Impact</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">1 layer</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">0.406</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Nodes still distinct</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Can identify different atoms</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">3 layers</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">0.995</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Nearly identical</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Losing atomic identity</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">5 layers</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">0.993</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Effectively same</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">No useful information</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">10 layers</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">1.000</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Complete collapse</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Model is useless</td>
        </tr>
    </table>
</div>

*As shown above, as the number of message-passing layers increases, node representations converge. Initially distinct feature vectors (left) become nearly indistinguishable after several layers (right), resulting in the loss of structural information. This phenomenon is known as **over-smoothing** and is a critical limitation of deep GNNs.*

**Interpretation**

As we can see, even at just 3 layers, the node embeddings become nearly identical. By 10 layers, the model has effectively lost all ability to distinguish individual atoms. This is the core issue of **over-smoothing**—deep GNNs can blur out meaningful structural differences.

<div style="background-color:#e8f5e9; padding:15px; border-radius:8px; margin:15px 0;">
    <h4>Solutions to Over-smoothing</h4>
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#a5d6a7;">
            <th style="padding:10px; border:1px solid #66bb6a;">Technique</th>
            <th style="padding:10px; border:1px solid #66bb6a;">How It Works</th>
            <th style="padding:10px; border:1px solid #66bb6a;">Implementation</th>
            <th style="padding:10px; border:1px solid #66bb6a;">Effectiveness</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;"><b>Residual Connections</b></td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">Skip connections preserve original features</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">x = x + GCN(x)</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">Very effective</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;"><b>Feature Concatenation</b></td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">Combine features from multiple layers</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">concat(x₁, x₂, ...)</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">Good for shallow nets</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;"><b>Batch Normalization</b></td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">Normalize features per layer</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">BatchNorm after GCN</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">Moderate help</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;"><b>Jumping Knowledge</b></td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">Aggregate all layer outputs</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">JK networks</td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">State-of-the-art</td>
        </tr>
    </table>
</div>

To mitigate this problem, modern GNNs use techniques like:
* **Residual connections** (skip connections that reintroduce raw input)
* **Feature concatenation from earlier layers**
* **Batch normalization or graph normalization**
* **Jumping knowledge networks** to combine representations from multiple layers

When working with molecular graphs, you should **choose the depth of your GNN carefully**. It should be **deep enough** to capture important substructures, but **not so deep** that you lose atomic-level details.

#### Interpretability in Molecular GNNs

Beyond the technical challenge of over-smoothing, GNNs face a critical issue of interpretability. When a model predicts that a molecule might be toxic or have specific properties, chemists need to understand which structural features drive that prediction. This "black box" nature of neural networks is particularly problematic in chemistry, where understanding structure-activity relationships is fundamental to rational drug design.

<div style="background-color:#f0f7ff; border:2px solid #1976d2; border-radius:10px; padding:20px; margin:20px 0;">
    <h4>Why Interpretability Matters in Chemistry</h4>
    
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#e3f2fd;">
            <th style="padding:10px; border:1px solid #90caf9;">Stakeholder</th>
            <th style="padding:10px; border:1px solid #90caf9;">Need</th>
            <th style="padding:10px; border:1px solid #90caf9;">Example</th>
            <th style="padding:10px; border:1px solid #90caf9;">Impact</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #90caf9;"><b>Medicinal Chemists</b></td>
            <td style="padding:10px; border:1px solid #90caf9;">Understand SAR<br><span style="color:#666; font-size:0.9em;">Structure-Activity Relationships</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">Which groups increase potency?</td>
            <td style="padding:10px; border:1px solid #90caf9;">Guide drug optimization</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #90caf9;"><b>Regulatory Bodies</b></td>
            <td style="padding:10px; border:1px solid #90caf9;">Safety justification<br><span style="color:#666; font-size:0.9em;">Why is it safe?</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">Explain toxicity predictions</td>
            <td style="padding:10px; border:1px solid #90caf9;">FDA approval</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #90caf9;"><b>Researchers</b></td>
            <td style="padding:10px; border:1px solid #90caf9;">Scientific insight<br><span style="color:#666; font-size:0.9em;">New mechanisms</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">Discover new pharmacophores</td>
            <td style="padding:10px; border:1px solid #90caf9;">Advance knowledge</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #90caf9;"><b>Industry</b></td>
            <td style="padding:10px; border:1px solid #90caf9;">Risk assessment<br><span style="color:#666; font-size:0.9em;">Confidence in predictions</span></td>
            <td style="padding:10px; border:1px solid #90caf9;">Why invest in this molecule?</td>
            <td style="padding:10px; border:1px solid #90caf9;">Resource allocation</td>
        </tr>
    </table>
</div>

Recent advances in GNN interpretability for molecular applications have taken several promising directions:

**Attention-Based Methods**: 

<div style="background-color:#e3f2fd; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#90caf9;">
            <th style="padding:10px; border:1px solid #42a5f5; text-align:center;" colspan="4">Attention-Based Interpretability</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>Method:</b><br>
                Graph Attention Networks<br>
                <span style="font-size:0.9em; color:#666;">GATs</span>
            </td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>How it works:</b><br>
                Learn importance weights<br>
                <span style="font-size:0.9em; color:#666;">α_ij for each edge</span>
            </td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>Visualization:</b><br>
                Highlight important bonds<br>
                <span style="font-size:0.9em; color:#666;">Thicker = more important</span>
            </td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">
                <b>Reference:</b><br>
                Veličković et al., 2017<br>
                <span style="font-size:0.9em; color:#666;">ICLR</span>
            </td>
        </tr>
    </table>
</div>

Graph Attention Networks (GATs) provide built-in interpretability through their attention mechanisms, allowing researchers to visualize which atoms or bonds the model considers most important for a given prediction [1,2]. This approach naturally aligns with chemical intuition about reactive sites and functional groups.

**Substructure-Based Explanations**: 

<div style="background-color:#f3e5f5; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#e1bee7;">
            <th style="padding:10px; border:1px solid #ba68c8; text-align:center;" colspan="4">Substructure Mask Explanation (SME)</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ce93d8; background-color:#ffffff;">
                <b>Innovation:</b><br>
                Fragment-based<br>
                <span style="font-size:0.9em; color:#666;">Not just atoms</span>
            </td>
            <td style="padding:10px; border:1px solid #ce93d8; background-color:#ffffff;">
                <b>Alignment:</b><br>
                Chemical intuition<br>
                <span style="font-size:0.9em; color:#666;">Functional groups</span>
            </td>
            <td style="padding:10px; border:1px solid #ce93d8; background-color:#ffffff;">
                <b>Application:</b><br>
                Toxicophore detection<br>
                <span style="font-size:0.9em; color:#666;">Find toxic substructures</span>
            </td>
            <td style="padding:10px; border:1px solid #ce93d8; background-color:#ffffff;">
                <b>Reference:</b><br>
                Nature Comms, 2023<br>
                <span style="font-size:0.9em; color:#666;">14, 2585</span>
            </td>
        </tr>
    </table>
</div>

The Substructure Mask Explanation (SME) method represents a significant advance by providing interpretations based on chemically meaningful molecular fragments rather than individual atoms or edges [3]. This approach uses established molecular segmentation methods to ensure explanations align with chemists' understanding, making it particularly valuable for identifying pharmacophores and toxicophores.

**Integration of Chemical Knowledge**: 

<div style="background-color:#e8f5e9; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#a5d6a7;">
            <th style="padding:10px; border:1px solid #66bb6a; text-align:center;" colspan="4">Pharmacophore-Integrated GNNs</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Concept:</b><br>
                Hierarchical modeling<br>
                <span style="font-size:0.9em; color:#666;">Multi-level structure</span>
            </td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Benefit 1:</b><br>
                Better performance<br>
                <span style="font-size:0.9em; color:#666;">Domain knowledge helps</span>
            </td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Benefit 2:</b><br>
                Natural interpretability<br>
                <span style="font-size:0.9em; color:#666;">Pharmacophore-level</span>
            </td>
            <td style="padding:10px; border:1px solid #81c784; background-color:#ffffff;">
                <b>Reference:</b><br>
                J Cheminformatics, 2022<br>
                <span style="font-size:0.9em; color:#666;">14, 49</span>
            </td>
        </tr>
    </table>
</div>

Recent work has shown that incorporating pharmacophore information hierarchically into GNN architectures not only improves prediction performance but also enhances interpretability by explicitly modeling chemically meaningful substructures [4]. This bridges the gap between data-driven learning and domain expertise.

**Gradient-Based Attribution**: 

<div style="background-color:#fff3e0; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#ffe082;">
            <th style="padding:10px; border:1px solid #ffc107; text-align:center;" colspan="4">SHAP for Molecular GNNs</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffd54f; background-color:#ffffff;">
                <b>Method:</b><br>
                SHapley values<br>
                <span style="font-size:0.9em; color:#666;">Game theory based</span>
            </td>
            <td style="padding:10px; border:1px solid #ffd54f; background-color:#ffffff;">
                <b>Advantage:</b><br>
                Rigorous foundation<br>
                <span style="font-size:0.9em; color:#666;">Additive features</span>
            </td>
            <td style="padding:10px; border:1px solid #ffd54f; background-color:#ffffff;">
                <b>Output:</b><br>
                Feature importance<br>
                <span style="font-size:0.9em; color:#666;">Per atom/bond</span>
            </td>
            <td style="padding:10px; border:1px solid #ffd54f; background-color:#ffffff;">
                <b>Reference:</b><br>
                Lundberg & Lee, 2017<br>
                <span style="font-size:0.9em; color:#666;">NeurIPS</span>
            </td>
        </tr>
    </table>
</div>

Methods like SHAP (SHapley Additive exPlanations) have been successfully applied to molecular property prediction, providing feature importance scores that help identify which molecular characteristics most influence predictions [5,6]. These approaches are particularly useful for understanding global model behavior across different molecular classes.

**Comparative Studies**: 

<div style="background-color:#ffebee; padding:15px; border-radius:8px; margin:15px 0;">
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#ffcdd2;">
            <th style="padding:10px; border:1px solid #ef9a9a; text-align:center;" colspan="4">GNNs vs Traditional Methods</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                <b>Aspect</b></td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                <b>GNNs</b></td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                <b>Descriptor-based</b></td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                <b>Recommendation</b></td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Performance</td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Often superior<br><span style="font-size:0.9em; color:#666;">Complex patterns</span></td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Good baseline<br><span style="font-size:0.9em; color:#666;">Well-understood</span></td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Task-dependent</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Interpretability</td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Challenging<br><span style="font-size:0.9em; color:#666;">Requires extra work</span></td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Built-in<br><span style="font-size:0.9em; color:#666;">Known features</span></td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Hybrid approach</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;">
                Reference</td>
            <td style="padding:10px; border:1px solid #ef9a9a; background-color:#ffffff;" colspan="3">
                Jiang et al., 2021, J Cheminformatics</td>
        </tr>
    </table>
</div>

Recent comparative studies have shown that while GNNs excel at learning complex patterns, traditional descriptor-based models often provide better interpretability through established chemical features, suggesting a potential hybrid approach combining both paradigms [6].

<div style="background-color:#f0f4c3; padding:15px; border-radius:8px; margin:20px 0;">
    <h4>The Future: Interpretable-by-Design</h4>
    <p>The field is moving toward interpretable-by-design architectures rather than post-hoc explanation methods. As noted by researchers, some medicinal chemists value interpretability over raw accuracy if a small sacrifice in performance can significantly enhance understanding of the model's reasoning [3]. This reflects a broader trend in molecular AI toward building systems that augment rather than replace human chemical intuition.</p>
    
    <table style="width:100%; border-collapse:collapse; margin-top:15px;">
        <tr style="background-color:#fff176;">
            <th style="padding:10px; border:1px solid #ffd600;">Design Principle</th>
            <th style="padding:10px; border:1px solid #ffd600;">Implementation</th>
            <th style="padding:10px; border:1px solid #ffd600;">Example</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Chemical hierarchy</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Multi-scale architectures</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Atom → Group → Molecule</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Explicit substructures</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Pharmacophore encoding</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">H-bond donors as nodes</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Modular predictions</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Separate property modules</td>
            <td style="padding:10px; border:1px solid #ffeb3b; background-color:#ffffff;">Solubility + Toxicity branches</td>
        </tr>
    </table>
</div>

#### Summary

<div style="background-color:#e3f2fd; padding:15px; border-radius:8px;">
    <h4>Key Takeaways: Challenges and Solutions</h4>
    
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#90caf9;">
            <th style="padding:10px; border:1px solid #42a5f5;">Challenge</th>
            <th style="padding:10px; border:1px solid #42a5f5;">Impact</th>
            <th style="padding:10px; border:1px solid #42a5f5;">Current Solutions</th>
            <th style="padding:10px; border:1px solid #42a5f5;">Future Directions</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;"><b>Over-smoothing</b></td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">Limits depth to 3-5 layers<br><span style="font-size:0.9em; color:#666;">Can't capture long-range</span></td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">• Residual connections<br>• Jumping knowledge<br>• Normalization</td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">Novel architectures<br><span style="font-size:0.9em; color:#666;">Beyond message passing</span></td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;"><b>Interpretability</b></td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">Low trust & adoption<br><span style="font-size:0.9em; color:#666;">Can't guide design</span></td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">• Attention visualization<br>• SHAP values<br>• Substructure masking</td>
            <td style="padding:10px; border:1px solid #64b5f6; background-color:#ffffff;">Interpretable-by-design<br><span style="font-size:0.9em; color:#666;">Chemical hierarchy</span></td>
        </tr>
    </table>
    
    <p style="margin-top:15px;"><b>The Path Forward:</b></p>
    <ul>
        <li><b>Balance accuracy with interpretability</b> - Sometimes 90% accuracy with clear explanations beats 95% black box</li>
        <li><b>Incorporate domain knowledge</b> - Chemical principles should guide architecture design</li>
        <li><b>Develop hybrid approaches</b> - Combine GNN power with traditional descriptor interpretability</li>
        <li><b>Focus on augmenting chemists</b> - Tools should enhance, not replace, human expertise</li>
    </ul>
</div>

The challenges facing molecular GNNs—over-smoothing and interpretability—are significant but surmountable. Over-smoothing limits the depth of networks we can effectively use, constraining the model's ability to capture long-range molecular interactions. Meanwhile, the interpretability challenge affects trust and adoption in real-world applications where understanding model decisions is crucial.

Current solutions include architectural innovations like residual connections to combat over-smoothing, and various interpretability methods ranging from attention visualization to substructure-based explanations. The key insight is that effective molecular AI systems must balance predictive power with chemical interpretability, ensuring that models not only make accurate predictions but also provide insights that align with and enhance human understanding of chemistry.

As the field progresses, the focus is shifting from purely accuracy-driven models to systems that provide transparent, chemically meaningful explanations for their predictions. This evolution is essential for GNNs to fulfill their promise as tools for accelerating molecular discovery and understanding.

##### References (3.3.4)

* Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. *International Conference on Learning Representations*.
* Yuan, H., Yu, H., Gui, S., & Ji, S. (2022). Explainability in graph neural networks: A taxonomic survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
* Chemistry-intuitive explanation of graph neural networks for molecular property prediction with substructure masking. (2023). *Nature Communications*, 14, 2585.
* Integrating concept of pharmacophore with graph neural networks for chemical property prediction and interpretation. (2022). *Journal of Cheminformatics*, 14, 52.
* Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.
* Jiang, D., Wu, Z., Hsieh, C. Y., Chen, G., Liao, B., Wang, Z., ... & Hou, T. (2021). Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models. *Journal of Cheminformatics*, 13(1), 1-23.
* 
---

### Section 3.3 – Quiz Questions

#### 1) Factual Questions

##### Question 1

What is the primary advantage of using Graph Neural Networks (GNNs) over traditional neural networks for molecular property prediction?

**A.** GNNs require less computational resources  
**B.** GNNs can directly process the graph structure of molecules  
**C.** GNNs always achieve higher accuracy than other methods  
**D.** GNNs work only with small molecules

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
GNNs can directly process molecules as graphs where atoms are nodes and bonds are edges, preserving the structural information that is crucial for determining molecular properties.
</details>

---

##### Question 2

In the message passing mechanism of GNNs, what happens during the aggregation step?

**A.** Node features are updated using a neural network  
**B.** Messages from neighboring nodes are combined  
**C.** Edge features are initialized  
**D.** The final molecular prediction is made

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
During aggregation, all incoming messages from neighboring nodes are combined (typically by summing or averaging) to form a single aggregated message for each node.
</details>

---

##### Question 3

Which of the following molecular representations is most suitable as input for a Graph Neural Network?

**A.** SMILES string directly as text  
**B.** 2D image of the molecular structure  
**C.** Graph with nodes as atoms and edges as bonds  
**D.** List of molecular descriptors only

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
GNNs are designed to work with graph-structured data where nodes represent atoms and edges represent chemical bonds, allowing the model to learn from the molecular connectivity.
</details>

---

##### Question 4

What is the "over-smoothing" problem in Graph Neural Networks?

**A.** The model becomes too complex to train  
**B.** Node representations become increasingly similar in deeper networks  
**C.** The model cannot handle large molecules  
**D.** Training takes too much time

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Over-smoothing occurs when deep GNNs make node representations increasingly similar across layers, losing the ability to distinguish between different atoms and their local environments.
</details>

---

#### 2) Conceptual Questions

##### Question 5

You want to build a GNN to predict molecular solubility (a continuous value). Which combination of pooling and output layers would be most appropriate?

**A.**

```python
# Mean pooling + regression output
x = global_mean_pool(x, batch)
output = nn.Linear(hidden_dim, 1)(x)
```

**B.**

```python
# Max pooling + classification output  
x = global_max_pool(x, batch)
output = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Softmax())(x)
```

**C.**

```python
# No pooling + multiple outputs
output = nn.Linear(hidden_dim, num_atoms)(x)
```

**D.**

```python
# Sum pooling + sigmoid output
x = global_add_pool(x, batch) 
output = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())(x)
```

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: A
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
For continuous property prediction (regression), we need to pool node features to get a molecular-level representation, then use a linear layer to output a single continuous value. Mean pooling is commonly used and effective for this purpose.
</details>

<details>
<summary>▶ Click to see code: Complete GNN architecture for solubility prediction</summary>
<pre><code class="language-python">
# Complete GNN for solubility prediction
class SolubilityGNN(nn.Module):
    def __init__(self, node_features, hidden_dim=64):
        super(SolubilityGNN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Pool to molecular level
        return self.predictor(x)        # Single continuous output
</code></pre>
</details>

---

##### Question 6

A chemist notices that their GNN model performs well on training molecules but poorly on a new set of structurally different compounds. What is the most likely cause and solution?

**A.** The model is too simple; add more layers  
**B.** The model suffers from distribution shift; collect more diverse training data  
**C.** The learning rate is too high; reduce it  
**D.** The model has too many parameters; reduce model size

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
This scenario describes distribution shift, where the model was trained on one chemical space but tested on a different one. The solution is to include more diverse molecular structures in the training data to improve generalization.
</details>

<details>
<summary>▶ Click to see code: Data augmentation for chemical space diversity</summary>
<pre><code class="language-python">
# Data augmentation to improve generalization
def augment_chemical_space(original_smiles_list):
    """Expand training data with structural diversity"""
    augmented_data = []
    
    for smiles in original_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        
        # Add original
        augmented_data.append(smiles)
        
        # Add different SMILES representations
        for _ in range(3):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_data.append(random_smiles)
    
    return augmented_data

# Use diverse training data from multiple chemical databases
diverse_training_data = combine_datasets([
    'drug_molecules.csv',
    'natural_products.csv', 
    'synthetic_compounds.csv'
])
</code></pre>
</details>
