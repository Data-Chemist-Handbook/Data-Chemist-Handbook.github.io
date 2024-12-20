---
title: 5. Molecular Property Optimization
author: Haomin
date: 2024-08-15
category: Jekyll
layout: post
---
dataset: PMO (https://arxiv.org/pdf/2206.12411)

In molecular property optimization, the vastness and complexity of chemical space pose a significant challenge. Chemists must navigate a nearly infinite number of possible molecular structures in search of those with the optimal properties. This task requires finding a balance between two key strategies: exploration—searching for entirely new or unexplored molecules that may offer better properties, and exploitation—focusing on refining and improving molecules already known to perform well.

## 5.1 Bayesian Optimization

### 5.1.1 Introduction to Bayesian Optimization

Bayesian optimization (BO) has emerged as a powerful tool for optimizing expensive-to-evaluate functions, particularly when working with complex molecular systems where running experiments or simulations can be costly and time-consuming. In molecular property optimization, where the goal is to find the optimal molecule with desired properties (such as binding affinity, stability, or solubility), traditional optimization methods like brute-force searches or gradient-based methods are often impractical. BO, by efficiently balancing exploration and exploitation of the molecular space, offers a practical solution for these optimization challenges.

Bayesian optimization (BO) achieves this balance through the use of probabilistic models that capture both the known information (exploitation) and the uncertainty about the molecular space (exploration). The surrogate model employed by BO, such as a Gaussian Process, allows the algorithm to make informed decisions about which molecules to evaluate next. The model predicts not only the expected property values of new molecules but also the uncertainty of those predictions.

This uncertainty is critical: BO deliberately chooses molecules with high uncertainty in unexplored regions to explore new areas of chemical space that might harbor better-performing candidates. At the same time, it also exploits known regions of the space by selecting molecules that are predicted to be near-optimal based on current knowledge. Acquisition functions like Expected Improvement (EI) or Upper Confidence Bound (UCB) mathematically combine these two strategies, ensuring that the optimization process doesn't get stuck in local optima (by only exploiting known regions) and also avoids wasting resources on exploring too many suboptimal candidates.

By constantly adjusting the trade-off between exploration and exploitation, BO allows chemists to efficiently search through a large molecular space, maximizing the chances of discovering optimal molecules while minimizing unnecessary experiments. This makes BO especially valuable in fields like drug discovery or materials science, where the cost of synthesizing and testing each molecule is high, and there is immense value in strategically selecting the most promising candidates.

### 5.1.2 Key Concepts of Bayesian Optimization

#### 5.1.2.1 Bayesian Inference

Bayesian optimization is grounded in Bayesian inference, which allows the model to update its predictions dynamically as new data becomes available. At the core of Bayesian inference is the concept of updating beliefs—that is, the model begins with a prior understanding of the objective function (in this case, a molecular property like solubility or binding affinity) and refines this understanding as each new molecule is evaluated. This is the main differentiator of Bayesian optimization from other algorithms, this highly adaptive and efficient method of exploring complex spaces.

For chemists, this means that as each molecular experiment or simulation is conducted, the surrogate model updates its predictions based on the new results. This continuous updating process enables informed decision-making: with each new piece of data, the model becomes better at distinguishing between promising and less promising regions of the molecular space. By incorporating prior knowledge and new data, Bayesian inference helps Bayesian optimization efficiently navigate complex and high-dimensional molecular landscapes.

#### 5.1.2.2 Surrogate Models

In Bayesian optimization, a surrogate model serves as a stand-in for the actual objective function, which might be costly or time-consuming to evaluate directly. This model approximates the molecular property of interest by constructing a mathematical representation based on the available data. The most commonly used surrogate model in BO is the Gaussian Process (GP), though other models like random forests or Bayesian neural networks can also be used.

Gaussian Processes are particularly valuable in molecular optimization due to their ability to not only predict the expected value of a molecular property but also provide an estimate of the uncertainty associated with that prediction. In Layman’s terms, GP essentially communicates whether or not it's confident if a molecule will perform well, or if it believes that it should continue exploring. This dual capability allows chemists to understand not only what is likely (the mean) but also how confident the model is in those predictions. The uncertainty helps balance exploration (sampling molecules with high uncertainty) and exploitation (sampling molecules likely to have desirable properties), as it identifies regions of the chemical space that are underexplored and potentially fruitful.

#### 5.1.2.3 Acquisition Functions

In Bayesian optimization, after we have our surrogate model (like the Gaussian Process), we need a strategy to decide which molecule to test next. This is where acquisition functions come into play.

You can think of acquisition functions as decision-making tools that help us choose the next candidate based on what we know so far. They evaluate the trade-off between two main ideas:

**Exploration:** This means looking at new, untested molecules that might give us useful information. It’s like exploring a new neighborhood; you might find hidden gems, but you also might find nothing exciting.

**Exploitation:** This means focusing on molecules that we already think will perform well based on our current knowledge. It’s like revisiting a favorite restaurant because you know you love the food.

The acquisition function helps find a balance between these two ideas. Here are a few common types of acquisition functions explained simply:

**Expected Improvement (EI):** This function looks at how much better we might expect a new molecule to perform compared to our current best result. It helps us find candidates that could give us the best improvements.

**Probability of Improvement (PI):** This function focuses on the chance that a new molecule will outperform our current best one. It’s all about finding candidates that have a high likelihood of being better.

**Upper Confidence Bound (UCB):** This function looks at both the predicted value of a molecule and the uncertainty around that prediction. It tends to favor molecules that are uncertain but could potentially have a high reward, encouraging exploration.

Acquisition functions are essential because they guide the optimization process. Instead of randomly picking the next molecule to test, they help us make smart decisions based on our current understanding. This way, we can efficiently search through a complex space, testing fewer molecules while still maximizing our chances of finding the best ones.

In summary, acquisition functions help us choose the next steps in our optimization journey by balancing the need to explore new possibilities and the desire to make the most of what we already know.


### 5.1.3 Bayesian Optimization for Molecular Property Optimization Shortcut

The EDBOplus library presents an easy shortcut option without having to understand the full process of Bayesian Optimization: https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/tutorials/1_CLI_example.ipynb

### 5.1.4 Bayesian Optimization for Molecular Property Optimization Full Manual Process

In the event where we may want to do something more complex, here is the full process laid out in an example looking to optimize logP values:

google colab: https://colab.research.google.com/drive/1uXjcu_bzygtia9xgEHN76xMmQtcV0sY-?usp=sharing

**Step 1: Install RDKit and Other Required Libraries**
<pre>
    <code class="python">
# Install RDKit in Google Colab
!pip install rdkit

# Import necessary libraries
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
from google.colab import files
import io
    </code>
</pre>

**Step 2: Download and Upload the BACE Dataset (Stage 1a) from Drug Design Data**

The **BACE dataset** is available on Drug Design Data: https://drugdesigndata.org/about/grand-challenge-4/bace

<pre>
    <code class="python">
# Load dataset
uploaded = files.upload()
print(uploaded.keys()) #verifying if getting a file error
data = pd.read_csv(io.BytesIO(uploaded['BACE_FEset_compounds_D3R_GC4.csv'])) #file name should match your download file
    </code>
</pre>

**Step 3: Understand the Dataset**

Use the head and columns function to understand what our dataset looks like.

<pre>
    <code class="python">
data.head()
    </code>
</pre>

<pre>
    <code class="python">
data.columns
    </code>
</pre>

**Step 4: Calculate Properties and Application to SMILES**


<pre>
   <code class = "python">
# Function to calculate LogP, Molecular Weight, and TPSA for each SMILES string, returning none if does not exist
def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        logP = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        return logP, mw, tpsa
    else:
        return None, None, None

# Apply the function to the SMILES column (SMILES contains complex organic molecules)
properties = data['smiles'].apply(calculate_properties)
data[['LogP', 'Molecular_Weight', 'TPSA']] = pd.DataFrame(properties.tolist(), index=data.index)
   </code>
</pre>

**Step 5: Select Target Properties for Optimization and Sample**
<pre>
   <code class = "python">
# Select the target property for optimization (e.g., LogP)
target_property = 'LogP'  # Change to 'Molecular_Weight' or 'TPSA' as desire
x_range = np.linspace(-2*np.pi, 2*np.pi, 100)  # Adjust the range based on your specific goals

# Sample x values and outputs from the target property
sample_x = np.array(range(len(data)))
sample_y = data[target_property].values
   </code>
</pre>

**Define Black Box Functions, Gaussian Process, and Bayesian Optimization**
<pre>
   <code class = "python">
def black_box_function(x):
    # Use the target property values directly
    return data[target_property].iloc[int(x)]  # Ensure x is an integer

def upper_confidence_bound(x, gp_model, beta):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    return y_pred + beta * y_std

# Initial setup for sample_x and sample_y
initial_x_values = np.array([0, 1, 2, 3])  # Example initial x values (replace with your actual initial values)
initial_y_values = np.array([10, 20, 15, 18])  # Corresponding initial y values (replace with your actual initial values)

sample_x = initial_x_values  # Initial sample_x values (ensure it is a 1D array)
sample_y = initial_y_values  # Corresponding initial sample_y values (same length as sample_x)

# Gaussian process with RBF kernel
kernel = RBF(length_scale=1.0)
gp_model = GaussianProcessRegressor(kernel=kernel)
num_iterations = 5

# Perform the optimization
for i in range(num_iterations):
    # Ensure sample_x and sample_y have the same length
    gp_model.fit(sample_x.reshape(-1, 1), sample_y)
    beta = 1.0 # change beta to lower exploration (lower value) or raise (higher value) to help the model focus more on exploiting the known best performing points
    ucb = upper_confidence_bound(sample_x, gp_model, beta)
    best_idx = np.argmax(ucb)

    # Append new data (ensure sample_x and sample_y stay the same length)
    new_x_value = sample_x[best_idx]  # Get the best value from sample_x
    new_y_value = black_box_function(new_x_value)  # Get the corresponding y value from the black box function

    # Append the new values to sample_x and sample_y
    sample_x = np.append(sample_x, new_x_value)
    sample_y = np.append(sample_y, new_y_value)

    # Plot the optimization progress for this iteration
    plt.plot(sample_x, sample_y, label=f"Iteration {i+1}")
    plt.xlabel("Compound Index")
    plt.ylabel(target_property)
    plt.legend()
    plt.show()  # Show the plot after each iteration

   </code>
</pre>

**Pinpointing the Optimal and Maximized LogP Value**
<pre>
   <code class = "python">
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def black_box_function(x):
    # Use the target property values directly
    return data[target_property].iloc[int(x)]  # Ensure x is an integer

def upper_confidence_bound(x, gp_model, beta):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    return y_pred + beta * y_std

# Initial setup for sample_x and sample_y
initial_x_values = np.array([0, 1, 2, 3])  # Example initial x values (replace with your actual initial values)
initial_y_values = np.array([10, 20, 15, 18])  # Corresponding initial y values (replace with your actual initial values)

sample_x = initial_x_values  # Initial sample_x values (ensure it is a 1D array)
sample_y = initial_y_values  # Corresponding initial sample_y values (same length as sample_x)

# Gaussian process with RBF kernel
kernel = RBF(length_scale=1.0)
gp_model = GaussianProcessRegressor(kernel=kernel)
num_iterations = 5

# Variables to track the best index and LogP
best_logP_value = -np.inf  # Start with a very low value
best_index = -1  # To store the index of the best LogP value

# Perform the optimization
for i in range(num_iterations):
    # Ensure sample_x and sample_y have the same length
    gp_model.fit(sample_x.reshape(-1, 1), sample_y)
    beta = 2.0
    ucb = upper_confidence_bound(sample_x, gp_model, beta)
    best_idx = np.argmax(ucb)

    # Append new data (ensure sample_x and sample_y stay the same length)
    new_x_value = sample_x[best_idx]  # Get the best value from sample_x
    new_y_value = black_box_function(new_x_value)  # Get the corresponding y value from the black box function

    # Append the new values to sample_x and sample_y
    sample_x = np.append(sample_x, new_x_value)
    sample_y = np.append(sample_y, new_y_value)

    # Track the best LogP value and its index
    if new_y_value > best_logP_value:
        best_logP_value = new_y_value
        best_index = new_x_value  # Track the index of the best LogP value

    # Plot the optimization progress
    plt.plot(sample_x, sample_y, label=f"Iteration {i+1}")

# After all iterations, print the best index and LogP value
print(f"Best LogP value: {best_logP_value} at index {best_index}")

plt.xlabel("Compound Index")
plt.ylabel(target_property)
plt.legend()
plt.show()
   </code>
</pre>

## 5.2 Reinforcement Learning

### 5.2.1: Overview of Reinforcement Learning

Reinforcement Learning (RL) is a subset of machine learning where an agent learns to make decisions by interacting with an environment. The agent takes actions in the environment and receives feedback in the form of rewards or penalties. The primary goal is to learn a strategy, or policy, that maximizes cumulative rewards over time.

In simple terms, imagine teaching a pet to perform tricks. You reward them with treats for performing the desired action (like sitting or rolling over) and may ignore or gently correct them when they do not comply. Over time, the pet learns which actions lead to the best outcomes.

### 5.2.2: The Markov Decision Process Framework

At the heart of reinforcement learning is the **Markov Decision Process (MDP)**, which provides a mathematical framework for modeling decision-making. An MDP consists of:

1. **States (S):** All possible situations the agent can find itself in.
2. **Actions (A):** All possible actions the agent can take in each state.
3. **Transition Function (T):** A function that describes the probability of moving from one state to another, given a specific action.
4. **Rewards (R):** A reward function that provides feedback to the agent, indicating the immediate reward received after taking an action in a state.
5. **Discount Factor (γ):** A factor between 0 and 1 that determines the importance of future rewards. A value close to 0 makes the agent focus on immediate rewards, while a value close to 1 makes it consider long-term rewards.

Mathematically, an MDP can be represented as a tuple (S, A, T, R, γ).

### 5.2.3: Conceptual Example in Chemistry

Let’s consider a simplified example involving a chemistry lab experiment. Imagine you are trying to determine the best temperature setting for a reaction to maximize yield.

- **States (S):** Different temperature settings (e.g., 20°C, 25°C, 30°C).
- **Actions (A):** Adjusting the temperature up or down.
- **Rewards (R):** The yield from the reaction at each temperature (e.g., 80% yield at 25°C, 60% yield at 20°C).

As you experiment, you would record the yield for each temperature and adjust your strategy based on the results. Over time, you learn which temperature yields the best results, similar to how an RL agent learns the best actions to take.

### 5.2.4: Environmental Types in Reinforcement Learning

Reinforcement learning environments can vary significantly. They can be categorized as:

1. **Fully Observable vs. Partially Observable:**
   - Fully observable environments provide complete information about the current state.
   - Partially observable environments may only provide limited information.

2. **Deterministic vs. Stochastic:**
   - Deterministic environments have predictable outcomes.
   - Stochastic environments involve randomness, making outcomes uncertain.

3. **Static vs. Dynamic:**
   - Static environments do not change while the agent is deliberating.
   - Dynamic environments may change based on the agent's actions or external factors.

### 5.2.5: Classifications of Reinforcement Learning

Reinforcement learning can be broadly divided into two categories:

#### 5.2.5.1: Model-Based Reinforcement Learning

In model-based reinforcement learning, the agent builds a model of the environment's dynamics. This allows the agent to simulate different actions and predict their outcomes before taking them. This approach can lead to faster learning since the agent can plan its actions based on the model.

#### 5.2.5.2: Model-Free Reinforcement Learning

Model-free reinforcement learning does not require a model of the environment. Instead, the agent learns directly from its experiences. This approach can be simpler to implement but may require more interactions with the environment to learn effectively. Popular model-free methods include Q-learning and policy gradient methods.

### 5.2.6: Understanding Value Functions

A value function estimates how good it is for the agent to be in a given state, considering the long-term reward. There are two primary types of value functions:

- **State Value Function (V):** Estimates the expected return from a state, following a specific policy.

  \[
  V(s) = E[R | s]
  \]

- **Action Value Function (Q):** Estimates the expected return from taking a specific action in a state and then following a specific policy.

  \[
  Q(s, a) = E[R | s, a]
  \]

Understanding value functions helps the agent evaluate the quality of its actions and adjust its strategy accordingly.

### 5.2.7: The Exploration-Exploitation Dilemma

One of the critical challenges in reinforcement learning is the trade-off between **exploration** and **exploitation**:

- **Exploration:** Trying new actions to discover their effects and potential rewards.
- **Exploitation:** Leveraging known actions that yield high rewards based on past experiences.

An effective RL algorithm must balance these two aspects to learn efficiently. For example, using an ε-greedy strategy, the agent occasionally selects a random action (exploration) while mostly selecting the best-known action (exploitation).

### 5.2.8: Policy-Based vs. Value-Based Approaches

Reinforcement learning can be classified into two main approaches based on how the agent learns:

- **Policy-Based Methods:** These methods directly optimize the policy, which maps states to actions. An example is the REINFORCE algorithm, which updates the policy based on the received rewards.

- **Value-Based Methods:** These methods estimate the value functions and derive the policy from them. Q-learning is a prominent example, where the agent updates its action value function based on the rewards received.

### 5.3.0: Practical Application in Chemistry

Let’s implement a simple reinforcement learning scenario using Python, where we’ll simulate adjusting the temperature for a chemical reaction to maximize yield. In this example, we'll use Q-learning as our learning method.

<pre>
    <code class="python">
import numpy as np
import random

# Define the environment
temperatures = [20, 25, 30]  # Possible temperature settings
yields = {20: 60, 25: 80, 30: 70}  # Yield for each temperature
num_actions = len(temperatures)
num_states = len(temperatures)
q_table = np.zeros((num_states, num_actions))  # Initialize Q-table

# Parameters for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Q-learning algorithm
for episode in range(1000):  # Number of episodes
    state = random.randint(0, num_states - 1)  # Start with a random state
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take action and observe reward
        yield_received = yields[temperatures[action]]
        
        # Update Q-value using the Bellman equation
        next_state = action  # In this simple case, next state is the action taken
        q_table[state, action] += alpha * (yield_received + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state  # Transition to the next state

# Output the learned Q-table
print("Learned Q-table:")
print(q_table)
    </code>
</pre>

This example demonstrates how a reinforcement learning agent can learn to maximize yield based on the temperature settings of a chemical reaction.

## 5.3 Genetic Algorithms

## 5.4 Generative models with conditions
