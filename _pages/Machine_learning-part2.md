---
title: 3-B. Machine Learning Models
author: Haomin
date: 2024-08-13
category: Jekyll
layout: post
---

## 3.2 Neural Network

A neural network is a computational model inspired by the neural structure of the human brain, designed to recognize patterns and learn from data. It consists of layers of interconnected nodes, or neurons, which process input data through weighted connections.

**Structure**: Neural networks typically include an input layer, one or more hidden layers, and an output layer. Each neuron in a layer is connected to neurons in the adjacent layers. The input layer receives data, the hidden layers transform this data through various operations, and the output layer produces the final prediction or classification.

**Functioning**: Data is fed into the network, where each neuron applies an activation function to its weighted sum of inputs. These activation functions introduce non-linearity, allowing the network to learn complex patterns. The output of the neurons is then passed to the next layer until the final prediction is made.

**Learning Process**: Neural networks learn through a process called training. During training, the network adjusts the weights of connections based on the error between its predictions and the actual values. This is achieved using algorithms like backpropagation and optimization techniques such as gradient descent, which iteratively updates the weights to minimize the prediction error.

### 3.2.1 Biological and Conceptual Foundations of Neural Networks

Neural networks are a class of machine learning models designed to learn patterns from data in order to make predictions or classifications. Their structure and behavior are loosely inspired by how the human brain processes information: through a large network of connected units that transmit signals to each other. Although artificial neural networks are mathematical rather than biological, this analogy provides a helpful starting point for understanding how they function.

**The Neural Analogy**

In a biological system, neurons receive input signals from other neurons, process those signals, and send output to downstream neurons. Similarly, an artificial neural network is composed of units called "neurons" or "nodes" that pass numerical values from one layer to the next. Each of these units receives inputs, processes them using a simple rule, and forwards the result.

This structure allows the network to build up an understanding of the input data through multiple layers of transformations. As information flows forward through the network—layer by layer—it becomes increasingly abstract. Early layers may focus on basic patterns in the input, while deeper layers detect more complex or chemically meaningful relationships.

**Layers of a Neural Network**

Neural networks are organized into three main types of layers:
- **Input Layer:** This is where the network receives the data. In chemistry applications, this might include molecular fingerprints, structural descriptors, or other numerical representations of a molecule.
- **Hidden Layers:** These are the internal layers where computations happen. The network adjusts its internal parameters to best relate the input to the desired output.
- **Output Layer:** This layer produces the final prediction. For example, it might output a predicted solubility value, a toxicity label, or the probability that a molecule is biologically active.

The depth (number of layers) and width (number of neurons in each layer) of a network affect its capacity to learn complex relationships.

**Why Chemists Use Neural Networks**

Many molecular properties—such as solubility, lipophilicity, toxicity, and biological activity—are influenced by intricate, nonlinear combinations of atomic features and substructures. These relationships are often difficult to express with a simple equation or rule.

Neural networks are especially useful in chemistry because:

- They can learn from large, complex datasets without needing detailed prior knowledge about how different features should be weighted.
- They can model nonlinear relationships, such as interactions between molecular substructures, electronic effects, and steric hindrance.
- They are flexible and can be applied to a wide range of tasks, from predicting reaction outcomes to screening drug candidates.

**How Learning Happens**

Unlike hardcoded rules, neural networks improve through a process of learning:
1. **Prediction:** The network uses its current understanding to make a guess about the output (e.g., predicting a molecule's solubility).
2. **Feedback:** It compares its prediction to the known, correct value.
3. **Adjustment:** It updates its internal parameters to make better predictions next time.

This process repeats over many examples, gradually improving the model's accuracy. Over time, the network can generalize—making reliable predictions on molecules it has never seen before.

### 3.2.2 The Structure of a Neural Network

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1xBQ6a24F6L45uOFkgML4E6z58jzsbRFe?usp=sharing)

The structure of a neural network refers to how its components are organized and how information flows from the input to the output. Understanding this structure is essential for applying neural networks to chemical problems, where numerical data about molecules must be transformed into meaningful predictions—such as solubility, reactivity, toxicity, or classification into chemical groups.

**Basic Building Blocks**

A typical neural network consists of three types of layers:

1. **Input Layer**

This is the first layer and represents the data you give the model. In chemistry, this might include:
- Molecular fingerprints (e.g., Morgan or ECFP4)
- Descriptor vectors (e.g., molecular weight, number of rotatable bonds)
- Graph embeddings (in more advanced architectures)

Each input feature corresponds to one "neuron" in this layer. The network doesn't modify the data here; it simply passes it forward.

2. **Hidden Layers**

These are the core of the network. They are composed of interconnected neurons that process the input data through a series of transformations. Each neuron:
- Multiplies each input by a weight (a learned importance factor)
- Adds the results together, along with a bias term
- Passes the result through an activation function to determine the output

Multiple hidden layers can extract increasingly abstract features. For example:
- First hidden layer: detects basic structural motifs (e.g., aromatic rings)
- Later hidden layers: model higher-order relationships (e.g., presence of specific pharmacophores)

The depth of a network (number of hidden layers) increases its capacity to model complex patterns, but also makes it more challenging to train.

3. **Output Layer**

This layer generates the final prediction. The number of output neurons depends on the type of task:
- One neuron for regression (e.g., predicting solubility)
- One neuron with a sigmoid function for binary classification (e.g., active vs. inactive)
- Multiple neurons with softmax for multi-class classification (e.g., toxicity categories)

**Activation Functions**

The activation function introduces non-linearity to the model. Without it, the network would behave like a linear regression model, unable to capture complex relationships. Common activation functions include:
- **ReLU (Rectified Linear Unit):** Returns 0 for negative inputs and the input itself for positive values. Efficient and widely used.
- **Sigmoid:** Squeezes inputs into the range (0,1), useful for probabilities.
- **Tanh:** Similar to sigmoid but outputs values between -1 and 1, often used in earlier layers.

These functions allow neural networks to model subtle chemical relationships, such as how a substructure might enhance activity in one molecular context but reduce it in another.

**Forward Pass: How Data Flows Through the Network**

The process of making a prediction is known as the forward pass. Here's what happens step-by-step:

1. Each input feature (e.g., molecular weight = 300) is multiplied by a corresponding weight.
2. The weighted inputs are summed and combined with a bias.
3. The result is passed through the activation function.
4. The output becomes the input to the next layer.

This process repeats until the final output is produced.

**Building a Simple Neural Network for Molecular Property Prediction**

Let's build a minimal neural network that takes molecular descriptors as input and predicts a continuous chemical property, such as aqueous solubility. We'll use TensorFlow and Keras.


```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Example molecular descriptors for 5 hypothetical molecules:
# Features: [Molecular Weight, LogP, Number of Rotatable Bonds]
X = np.array([
    [180.1, 1.2, 3],
    [310.5, 3.1, 5],
    [150.3, 0.5, 2],
    [420.8, 4.2, 8],
    [275.0, 2.0, 4]
])

# Target values: Normalized aqueous solubility
y = np.array([0.82, 0.35, 0.90, 0.20, 0.55])

# Define a simple feedforward neural network
model = models.Sequential([
    layers.Input(shape=(3,)),              # 3 input features per molecule
    layers.Dense(8, activation='relu'),    # First hidden layer
    layers.Dense(4, activation='relu'),    # Second hidden layer
    layers.Dense(1)                        # Output layer (regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Mean Squared Error for regression

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Predict on new data
new_molecule = np.array([[300.0, 2.5, 6]])
predicted_solubility = model.predict(new_molecule)
print("Predicted Solubility:", predicted_solubility[0][0])
```

**Results**
```python
Predicted Solubility: 13.366545
```

**What This Code Does:**
- Inputs are numerical molecular descriptors (easy for chemists to relate to).
- The model learns a pattern from these descriptors to predict solubility.
- Layers are built exactly as explained: input → hidden (ReLU) → output.
- The output is a single continuous number, suitable for regression tasks.

**Practice Problem 3: Neural Network Warm-Up**

Using the logic from the code above:
1. Replace the input features with the following descriptors:
    - [350.2, 3.3, 5], [275.4, 1.8, 4], [125.7, 0.2, 1]
2. Create a new NumPy array called X_new with those values.
3. Use the trained model to predict the solubility of each new molecule.
4. Print the outputs with a message like:
    "Predicted solubility for molecule 1: 0.67"
   
```python
# Step 1: Create new molecular descriptors for prediction
X_new = np.array([
    [350.2, 3.3, 5],
    [275.4, 1.8, 4],
    [125.7, 0.2, 1]
])

# Step 2: Use the trained model to predict solubility
predictions = model.predict(X_new)

# Step 3: Print each result with a message
for i, prediction in enumerate(predictions):
    print(f"Predicted solubility for molecule {i + 1}: {prediction[0]:.2f}")
```

**Discussion: What Did We Just Do?**

In this practice problem, we used a trained neural network to predict the solubility of three new chemical compounds based on simple molecular descriptors. Each molecule was described using three features:
1. Molecular weight
2. LogP (a measure of lipophilicity)
3. Number of rotatable bonds

The model, having already learned patterns from prior data during training, applied its internal weights and biases to compute a prediction for each molecule.

```python
Predicted solubility for molecule 1: 0.38  
Predicted solubility for molecule 2: 0.55  
Predicted solubility for molecule 3: 0.91
```

These values reflect the model's confidence in how soluble each molecule is, with higher numbers generally indicating better solubility. While we don't yet know how the model arrived at these exact numbers (that comes in the next section), this exercise demonstrates a key advantage of neural networks:
- Once trained, they can generalize to unseen data—making predictions for new molecules quickly and efficiently.

### 3.2.3 How Neural Networks Learn: Backpropagation and Loss Functions

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1pTOPZNOpcDnMm0SrcdgGQAAfEpT5TvZF?usp=sharing)

In the previous section, we saw how a neural network can take molecular descriptors as input and generate predictions, such as aqueous solubility. However, this raises an important question: **how does the network learn to make accurate predictions in the first place?** The answer lies in two fundamental concepts: the **loss function** and **backpropagation**.

#### Loss Function: Measuring the Error

The **loss function** is a mathematical expression that quantifies how far off the model's predictions are from the actual values. It acts as a feedback mechanism—telling the network how well or poorly it's performing.

In regression tasks like solubility prediction, a common loss function is **Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

Where:
- $\hat{y}_i$ is the predicted solubility
- $y_i$ is the true solubility  
- $n$ is the number of samples

MSE penalizes larger errors more severely than smaller ones, which is especially useful in chemical property prediction where large prediction errors can have significant consequences.

#### Gradient Descent: Minimizing the Loss

Once the model calculates the loss, it needs to adjust its internal weights to reduce that loss. This optimization process is called **gradient descent**.

Gradient descent updates the model's weights in the opposite direction of the gradient of the loss function:

$$
w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial \text{Loss}}{\partial w}
$$

Where:
- $w$ is a weight in the network  
- $\alpha$ is the **learning rate**, a small scalar that determines the step size

This iterative update helps the model gradually "descend" toward a configuration that minimizes the prediction error.

#### Backpropagation: Updating the Network

**Backpropagation** is the algorithm that computes how to adjust the weights.

1. It begins by computing the prediction and measuring the loss.
2. Then, it calculates how much each neuron contributed to the final error by applying the **chain rule** from calculus.
3. Finally, it adjusts all weights by propagating the error backward from the output layer to the input layer.

Over time, the network becomes better at associating input features with the correct output properties.

#### Intuition for Chemists

Think of a chemist optimizing a synthesis route. After a failed reaction, they adjust parameters (temperature, solvent, reactants) based on what went wrong. With enough trials and feedback, they achieve better yields.

A neural network does the same—after each "trial" (training pass), it adjusts its internal settings (weights) to improve its "yield" (prediction accuracy) the next time.

**Visualizing Loss Reduction During Training**

This code demonstrates how a simple neural network learns over time by minimizing error through backpropagation and gradient descent. It also visualizes the loss curve to help you understand how training progresses.

```python
# 3.2.3 Example: Visualizing Loss Reduction During Training

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simulated training data: [molecular_weight, logP, rotatable_bonds]
X_train = np.array([
    [350.2, 3.3, 5],
    [275.4, 1.8, 4],
    [125.7, 0.2, 1],
    [300.1, 2.5, 3],
    [180.3, 0.5, 2]
])

# Simulated solubility labels (normalized between 0 and 1)
y_train = np.array([0.42, 0.63, 0.91, 0.52, 0.86])

# Define a simple neural network
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Regression output

# Compile the model using MSE (Mean Squared Error) loss
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model and record loss values
history = model.fit(X_train, y_train, epochs=100, verbose=0)

# Plot the training loss over time
plt.plot(history.history['loss'])
plt.title('Loss Reduction During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.show()
```

**This example demonstrates:**
- How the network calculates and minimizes the loss function (MSE)
- How backpropagation adjusts weights over time
- How loss consistently decreases with each epoch

**Practice Problem: Observe the Learning Curve**

Reinforce the concepts of backpropagation and gradient descent by modifying the model to exaggerate or dampen learning behavior.
1. Change the optimizer from "adam" to "sgd" and observe how the loss reduction changes.
2. Add validation_split=0.2 to model.fit() to visualize both training and validation loss.
3. Plot both loss curves using matplotlib.

```python
# Add validation and switch optimizer
model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

You should observe:
1. Slower convergence when using SGD vs. Adam.
2. Validation loss potentially diverging if overfitting begins.

### 3.2.4 Activation Functions

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1ycFBgKmtaej3WhiOnwpSH0iYqEPqXRnG?usp=sharing)

Activation functions are a key component of neural networks that allow them to model complex, non-linear relationships between inputs and outputs. Without activation functions, no matter how many layers we add, a neural network would essentially behave like a linear model. For chemists, this would mean failing to capture the non-linear relationships between molecular descriptors and properties such as solubility, reactivity, or binding affinity.

**What Is an Activation Function?**

An activation function is applied to the output of each neuron in a hidden layer. It determines whether that neuron should "fire" (i.e., pass information to the next layer) and to what degree.

Think of it like a valve in a chemical reaction pathway: the valve can allow the signal to pass completely, partially, or not at all—depending on the condition (input value). This gating mechanism allows neural networks to build more expressive models that can simulate highly non-linear chemical behavior.

**Common Activation Functions (with Intuition)**

Here are the most widely used activation functions and how you can interpret them in chemical modeling contexts:

**1. ReLU (Rectified Linear Unit)**

$$
\text{ReLU}(x) = \max(0,x)
$$

**Behavior**: Passes positive values as-is; blocks negative ones.  
**Analogy**: A pH-dependent gate that opens only if the environment is basic (positive).  
**Use**: Fast to compute; ideal for hidden layers in large models.

**2. Sigmoid**

$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

**Behavior**: Maps input to a value between 0 and 1.  
**Analogy**: Represents probability or confidence — useful when you want to interpret the output as "likelihood of solubility" or "chance of toxicity".  
**Use**: Often used in the output layer for binary classification.

**3. Tanh (Hyperbolic Tangent)**

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Behavior**: Outputs values between -1 and 1, centered around 0.  
**Analogy**: Models systems with directionality — such as positive vs. negative binding affinity.  
**Use**: Sometimes preferred over sigmoid in hidden layers.

**Why Are They Important**

Without activation functions, neural networks would be limited to computing weighted sums—essentially doing linear algebra. This would be like trying to model the melting point of a compound using only molecular weight: too simplistic for real-world chemistry.

Activation functions allow networks to "bend" input-output mappings, much like how a catalyst changes the energy profile of a chemical reaction.

**Comparing ReLU and Sigmoid Activation Functions**

This code visually compares how ReLU and Sigmoid behave across a range of inputs. Understanding the shapes of these activation functions helps chemists choose the right one for a neural network layer depending on the task (e.g., regression vs. classification).

```python
# 3.2.4 Example: Comparing ReLU vs Sigmoid Activation Functions

import numpy as np
import matplotlib.pyplot as plt

# Define ReLU and Sigmoid activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input range
x = np.linspace(-10, 10, 500)

# Compute function outputs
relu_output = relu(x)
sigmoid_output = sigmoid(x)

# Plot the functions
plt.figure(figsize=(10, 6))
plt.plot(x, relu_output, label='ReLU', linewidth=2)
plt.plot(x, sigmoid_output, label='Sigmoid', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title('Activation Function Comparison: ReLU vs Sigmoid')
plt.xlabel('Input (x)')
plt.ylabel('Activation Output')
plt.legend()
plt.grid(True)
plt.show()
```

**This example demonstrates:**
- ReLU outputs 0 for any negative input and increases linearly for positive inputs. This makes it ideal for deep layers in large models where speed and sparsity are priorities.
- Sigmoid smoothly maps all inputs to values between 0 and 1. This is useful for binary classification tasks, such as predicting whether a molecule is toxic or not.
- Why this matters in chemistry: Choosing the right activation function can affect whether your neural network correctly learns properties like solubility, toxicity, or reactivity. For instance, sigmoid may be used in the output layer when predicting probabilities, while ReLU is preferred in hidden layers to retain training efficiency.

### 3.2.5 Training a Neural Network for Chemical Property Prediction

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1IhwOUxG9xp8hEt9nkR04u1wTb3hr3Gal?usp=sharing)

In the previous sections, we explored how neural networks are structured and how they learn. In this final section, we'll put everything together by training a neural network on a small dataset of molecules to predict aqueous solubility — a property of significant importance in drug design and formulation.

Rather than using high-level abstractions, we'll walk through the full training process: from preparing chemical data to building, training, evaluating, and interpreting a neural network model.

**Chemical Context**

Solubility determines how well a molecule dissolves in water, which affects its absorption and distribution in biological systems. Predicting this property accurately can save time and cost in early drug discovery. By using features like molecular weight, lipophilicity (LogP), and number of rotatable bonds, we can teach a neural network to approximate this property from molecular descriptors.

**Step-by-Step Training Example**

Goal: Predict normalized solubility values from 3 molecular descriptors:
- Molecular weight
- LogP
- Number of rotatable bonds
    
```python
# 3.2.5 Example: Training a Neural Network for Solubility Prediction

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Simulated chemical data
X = np.array([
    [350.2, 3.3, 5],
    [275.4, 1.8, 4],
    [125.7, 0.2, 1],
    [300.1, 2.5, 3],
    [180.3, 0.5, 2],
    [410.0, 4.1, 6],
    [220.1, 1.2, 3],
    [140.0, 0.1, 1]
])
y = np.array([0.42, 0.63, 0.91, 0.52, 0.86, 0.34, 0.70, 0.95])  # Normalized solubility

# Step 2: Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Step 4: Build the neural network
model = Sequential()
model.add(Dense(16, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for regression (normalized range)

# Step 5: Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=100, verbose=0)

# Step 6: Evaluate performance
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss:.4f}")

# Step 7: Plot training loss
plt.plot(history.history['loss'])
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()
```

**Interpreting the Results**
- The network gradually learns to predict solubility based on three molecular features.
- The loss value shows the mean squared error on the test set—lower values mean better predictions.
- The loss curve demonstrates whether the model is converging (flattening loss) or struggling (oscillating loss).

**Summary**

This section demonstrated how a basic neural network can be trained on molecular descriptors to predict solubility. While our dataset was small and artificial, the same principles apply to real-world cheminformatics datasets.

You now understand:
- How to process input features from molecules
- How to build and train a simple feedforward neural network
- How to interpret loss, predictions, and model performance

This hands-on foundation prepares you to tackle more complex models like convolutional and graph neural networks in the next sections.

---

### Section 3.2 – Quiz Questions

#### 1) Factual Questions

##### Question 1
Which of the following best describes the role of the hidden layers in a neural network predicting chemical properties?

**A.** They store the molecular structure for visualization.  
**B.** They transform input features into increasingly abstract representations.  
**C.** They calculate the final solubility or toxicity score directly.  
**D.** They normalize the input data before processing begins.

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Hidden layers apply weights, biases, and activation functions to extract increasingly complex patterns (e.g., substructures, steric hindrance) from the input molecular data.
</details>

---

##### Question 2
Suppose you're predicting aqueous solubility using a neural network. Which activation function in the hidden layers would be most suitable to introduce non-linearity efficiently, especially with large chemical datasets?

**A.** Softmax  
**B.** Linear  
**C.** ReLU  
**D.** Sigmoid

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
ReLU is widely used in hidden layers for its computational efficiency and ability to handle vanishing gradient problems in large datasets.
</details>

---

##### Question 3
In the context of molecular property prediction, which of the following sets of input features is most appropriate for the input layer of a neural network?

**A.** IUPAC names and structural diagrams  
**B.** Raw SMILES strings and melting points as text  
**C.** Numerical descriptors like molecular weight, LogP, and rotatable bonds  
**D.** Hand-drawn chemical structures and reaction mechanisms

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Neural networks require numerical input. Molecular descriptors are quantifiable features that encode structural, electronic, and steric properties.
</details>

---

##### Question 4
Your neural network performs poorly on new molecular data but does very well on training data. Which of the following is most likely the cause?

**A.** The model lacks an output layer  
**B.** The training set contains irrelevant descriptors  
**C.** The network is overfitting due to too many parameters  
**D.** The input layer uses too few neurons

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Overfitting occurs when a model memorizes the training data but fails to generalize. This is common in deep networks with many parameters and not enough regularization or data diversity.
</details>

---

#### 2) Conceptual Questions

##### Question 5
You are building a neural network to predict binary activity (active vs inactive) of molecules based on three features: [Molecular Weight, LogP, Rotatable Bonds].  
Which code correctly defines the output layer for this classification task?

**A.** layers.Dense(1)  
**B.** layers.Dense(1, activation='sigmoid')  
**C.** layers.Dense(2, activation='relu')  
**D.** layers.Dense(3, activation='softmax')

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
For binary classification, you need a single neuron with a sigmoid activation function to output a probability between 0 and 1.
</details>

---

##### Question 6
Why might a chemist prefer a neural network over a simple linear regression model for predicting molecular toxicity?

**A.** Neural networks can run faster than linear models.  
**B.** Toxicity is not predictable using any mathematical model.  
**C.** Neural networks can model nonlinear interactions between substructures.  
**D.** Neural networks use fewer parameters and are easier to interpret.

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Chemical toxicity often arises from complex, nonlinear interactions among molecular features—something neural networks can capture but linear regression cannot.
</details>
