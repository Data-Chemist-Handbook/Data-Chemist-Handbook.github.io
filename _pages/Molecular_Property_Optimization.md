---
title: 5. Molecular Property Optimization
author: Haomin
date: 2024-08-15
category: Jekyll
layout: post
---

Dataset reference: PMO Benchmark - [Click Here](https://arxiv.org/pdf/2206.12411)

In drug discovery and materials design, chemists often face the daunting challenge of selecting the next best molecule to synthesize or simulate from an enormous chemical space—estimated to contain over 10e60 possible organic compounds. With limited time and experimental resources, evaluating every candidate molecule isn't feasible. Instead, chemists must strike a balance between exploration—testing new, untried molecules that might perform well—and exploitation—focusing on refining promising structures that are already known to behave favorably.

This is the core of molecular property optimization: making intelligent decisions about which molecules to prioritize, especially when targeting properties like solubility, binding affinity, lipophilicity (LogP), or synthetic accessibility. Computational methods have become indispensable tools in this pursuit, and one of the most effective frameworks for tackling this problem is Bayesian Optimization.

## 5.1 Bayesian Optimization

### 5.1.1 Introduction to Bayesian Optimization

Let's say you've synthesized a small batch of molecules and measured their solubility. A few candidates show promise, but you're working with limited time, budget, or access to high-throughput screening. You now face a dilemma: Should you modify your current leads slightly and improve them (exploit), or try something radically different (explore)? This is where Bayesian Optimization (BO) becomes a game-changer for chemists.

Bayesian Optimization is a strategy for efficiently finding the optimal molecule when evaluations are expensive—whether that means physical synthesis, simulation, or time-consuming quantum calculations. BO has been successfully applied to problems like optimizing drug-likeness, reaction yields, or physicochemical properties such as LogP.

What makes BO effective is its ability to learn as it goes, constantly updating a model of chemical space based on each new experiment or simulation. It balances exploration and exploitation by using a surrogate model—a kind of simplified approximation of reality—that estimates how well different molecules might perform.

The most commonly used surrogate in chemistry applications is the Gaussian Process (GP). This model not only predicts how promising a candidate is (expected property value), but also how uncertain that prediction is. For instance, it may estimate that a given molecule has a likely LogP of 3.2—but with a wide confidence interval because that region of chemical space is poorly understood.

This dual prediction (value + uncertainty) is what enables BO to be smart about its next move. It selects the next molecule to evaluate using an acquisition function, which combines the predicted performance and the uncertainty. Two commonly used acquisition functions in chemical optimization are:
- **Expected Improvement (EI):** Prefers molecules that could significantly outperform current best candidates.
- **Upper Confidence Bound (UCB):** Selects molecules with both high predicted value and high uncertainty, encouraging exploration.

By iteratively updating the surrogate model and guiding where to sample next, BO helps chemists avoid wasteful blind searches. Instead of evaluating thousands of compounds randomly, BO steers you toward the most promising and informative candidates—saving time, cost, and effort.

**Chemist's Insight:** Think of BO like a lab assistant that remembers every result you've seen and suggests the next experiment with the highest chance of success or insight.

Bayesian Optimization has become a cornerstone of modern computational chemistry pipelines, especially in fields where wet-lab validation is slow or costly. Whether you're designing a new drug molecule or optimizing materials for energy storage, BO can dramatically speed up the discovery process by helping you make smarter choices, faster.

### 5.1.2 Key Concepts of Bayesian Optimization

Bayesian Optimization may sound complex, but its power lies in just a few fundamental ideas. Together, these concepts enable chemists to prioritize which molecules to test next, all while minimizing wasted experiments. In this section, we'll break down the three core components of BO—**Bayesian inference, surrogate models, and acquisition functions**—with examples grounded in molecular property prediction.

#### 5.1.2.1 Bayesian Inference

At the heart of Bayesian Optimization is Bayesian inference—a framework for updating beliefs based on new evidence. Chemists already do this intuitively. For example, if a compound with a certain functional group shows high solubility, you're more likely to try similar ones next. BO formalizes this process mathematically.

In BO, we start with a prior belief about how molecules behave (e.g., which structures are likely to have high LogP or low toxicity). As we evaluate each molecule—whether through lab testing or simulation—we collect data and update our model to form a posterior belief: a more informed estimate of which molecules might perform best.

**Chemist's Insight:** Bayesian inference is like refining your mental model after each experiment—except here, the computer does it quantitatively and consistently.

This constant updating is what makes BO highly efficient. It ensures that each new test is more informed than the last, helping chemists navigate complex chemical spaces with fewer missteps.

#### 5.1.2.2 Surrogate Models

When synthesizing a molecule or running a simulation takes hours or days, we can't afford to try every candidate blindly. That's where surrogate models come in. They act as fast approximations of your real experiments.

In Bayesian Optimization, the most popular surrogate model is the Gaussian Process (GP). You can think of it like a "smart guesser" that:
- Predicts how well a molecule might perform (e.g., expected solubility)
- Tells you how confident it is in that prediction (e.g., uncertainty in underexplored regions)

This is especially helpful in chemistry, where relationships between structure and property are often non-linear and hard to generalize.

**Example:** A GP might estimate that a new compound has a predicted LogP of 2.1 ± 0.8. That wide margin tells us the model isn't very confident—maybe because similar molecules haven't been tested yet.

This built-in uncertainty allows the optimizer to identify gaps in knowledge and target them strategically. By modeling both what we know and what we don't, surrogate models help prioritize molecules that either have strong potential or will teach us something new.

#### 5.1.2.3 Acquisition Functions

Once the surrogate model is built, we still need a way to decide which molecule to evaluate next. That's the job of the acquisition function—a mathematical tool that scores all potential candidates based on their predicted value and uncertainty.

In simple terms, acquisition functions help balance two goals:
- **Exploration:** Try something new to learn more about chemical space.
- **Exploitation:** Focus on known regions that already show promising results.

**Analogy:** It's like choosing between trying a new restaurant or going back to one you already love. Exploration may lead to a hidden gem, while exploitation gives you a reliable experience.

Here are three commonly used acquisition functions in chemistry applications:
- **Expected Improvement (EI):** Estimates how much better a new candidate could be compared to the best one tested so far. It balances optimism and realism.
- **Probability of Improvement (PI):** Prioritizes candidates that are likely to beat the current best, even if the improvement is small.
- **Upper Confidence Bound (UCB):** Favors candidates with high uncertainty and high predicted value. It's bold and curious, often uncovering new "hot spots" in unexplored regions.

Each acquisition function has its strengths, and the best choice often depends on how cautious or adventurous you want the search to be.

**Chemist's Tip:** UCB is especially useful when you're unsure about the landscape—like optimizing a new reaction with limited precedent. EI is better once you have solid leads and want to fine-tune results.

**In Summary**
Bayesian Optimization works because it combines:
- A Bayesian mindset that updates knowledge as experiments are run
- A surrogate model that predicts both value and uncertainty
- An acquisition function that strategically guides what to test next

Together, these tools allow chemists to move through chemical space more intelligently—testing fewer molecules, making better decisions, and accelerating discovery.

### 5.1.3 Bayesian Optimization for Molecular Property Optimization Shortcut

The EDBOplus library presents an easy shortcut option without having to understand the full process of Bayesian Optimization: [Click Here](https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/tutorials/1_CLI_example.ipynb)

### 5.1.4 Bayesian Optimization for Molecular Property Optimization Full Manual Process

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/1uXjcu_bzygtia9xgEHN76xMmQtcV0sY-?usp=sharing)

In the event where we may want to do something more complex, here is the full process laid out in an example looking to optimize logP values:

**Step 1: Install RDKit and Other Required Libraries**
```python
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
```

**Step 2: Download and Upload the BACE Dataset (Stage 1a) from Drug Design Data**

The **BACE dataset** is available on Drug Design Data: [Click Here](https://drugdesigndata.org/about/grand-challenge-4/bace)

```python
# Load dataset
uploaded = files.upload()
print(uploaded.keys()) #verifying if getting a file error
data = pd.read_csv(io.BytesIO(uploaded['BACE_FEset_compounds_D3R_GC4.csv'])) #file name should match your download file
```

**Step 3: Understand the Dataset**

Use the head and columns function to understand what our dataset looks like.

```python
data.head()
```

```python
data.columns
```

**Step 4: Calculate Properties and Application to SMILES**

```python
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
```

**Step 5: Select Target Properties for Optimization and Sample**
```python
# Select the target property for optimization (e.g., LogP)
target_property = 'LogP'  # Change to 'Molecular_Weight' or 'TPSA' as desire
x_range = np.linspace(-2*np.pi, 2*np.pi, 100)  # Adjust the range based on your specific goals

# Sample x values and outputs from the target property
sample_x = np.array(range(len(data)))
sample_y = data[target_property].values
```

**Define Black Box Functions, Gaussian Process, and Bayesian Optimization**
```python
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
```

**Pinpointing the Optimal and Maximized LogP Value**
```python
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
```

---

### Section 5.1 – Quiz Questions

#### 1) Factual Questions

##### Question 1
What is the main advantage of Bayesian Optimization when applied to molecular property optimization?

**A.** It runs simulations faster by parallelizing them across GPUs  
**B.** It avoids the need for molecular descriptors  
**C.** It balances exploring unknown molecules and exploiting promising candidates  
**D.** It guarantees the global optimum after one evaluation

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Bayesian Optimization balances exploration of uncertain regions with exploitation of known good regions, making it efficient for expensive-to-evaluate molecular problems.
</details>

---

##### Question 2
What role does the Gaussian Process play in Bayesian Optimization?

**A.** It determines which chemical reactions are exothermic  
**B.** It predicts both expected values and uncertainty for new candidates  
**C.** It performs clustering to group similar molecules  
**D.** It selects which molecules to synthesize based on solubility alone

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: A Gaussian Process is a surrogate model that estimates both the expected value (mean) and the uncertainty (standard deviation) of a molecule's predicted property.
</details>

---

##### Question 3
Which of the following is an acquisition function used in Bayesian Optimization?

**A.** Mean Squared Error  
**B.** Pearson Correlation  
**C.** Upper Confidence Bound (UCB)  
**D.** ReLU Activation

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Upper Confidence Bound (UCB) is one of several acquisition functions that guides Bayesian Optimization by considering both predicted value and uncertainty.
</details>

---

##### Question 4
In the context of molecular optimization, what is the main purpose of an acquisition function?

**A.** To visualize molecular structures in 3D  
**B.** To convert SMILES to descriptors  
**C.** To determine the next molecule to evaluate  
**D.** To normalize the dataset

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Acquisition functions help choose the next point (molecule) to evaluate by balancing exploration and exploitation using the surrogate model's predictions.
</details>

---

#### 2) Conceptual Questions

##### Question 5
Why might a chemist choose Bayesian Optimization instead of grid search when optimizing molecular binding affinity?

**A.** Grid search always converges to a suboptimal result  
**B.** Bayesian Optimization can suggest new experiments based on prior results  
**C.** Grid search cannot be used on numerical data  
**D.** Bayesian Optimization doesn't need labeled training data

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Bayesian Optimization uses Bayesian inference to update a surrogate model with each new result, allowing it to strategically propose new evaluations based on the evolving understanding of chemical space.
</details>

---

##### Question 6
Which of the following best describes the exploration vs. exploitation trade-off in Bayesian Optimization?

**A.** Exploration only occurs once before exploitation begins  
**B.** Exploitation selects high-uncertainty candidates; exploration picks low-uncertainty ones  
**C.** Exploration chooses unknown regions; exploitation uses known good regions  
**D.** They are interchangeable and used randomly

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Exploration focuses on high-uncertainty regions that could reveal new optimal molecules, while exploitation focuses on areas already known to perform well.
</details>

---

## 5.2 Reinforcement Learning

### 5.2.1 Overview of Reinforcement Learning

Reinforcement Learning (RL) is a subset of machine learning where an agent learns to make decisions by interacting with an environment. The agent takes actions in the environment and receives feedback in the form of rewards or penalties. The primary goal is to learn a strategy, or policy, that maximizes cumulative rewards over time.

In simple terms, imagine teaching a pet to perform tricks. You reward them with treats for performing the desired action (like sitting or rolling over) and may ignore or gently correct them when they do not comply. Over time, the pet learns which actions lead to the best outcomes.

### 5.2.2 The Markov Decision Process Framework

At the heart of reinforcement learning is the **Markov Decision Process (MDP)**, which provides a mathematical framework for modeling decision-making. An MDP consists of:

1. **States (S):** All possible situations the agent can find itself in.
2. **Actions (A):** All possible actions the agent can take in each state.
3. **Transition Function (T):** A function that describes the probability of moving from one state to another, given a specific action.
4. **Rewards (R):** A reward function that provides feedback to the agent, indicating the immediate reward received after taking an action in a state.
5. **Discount Factor (γ):** A factor between 0 and 1 that determines the importance of future rewards. A value close to 0 makes the agent focus on immediate rewards, while a value close to 1 makes it consider long-term rewards.

Mathematically, an MDP can be represented as a tuple (S, A, T, R, γ).

### 5.2.3 Conceptual Example in Chemistry

Let's consider a simplified example involving a chemistry lab experiment. Imagine you are trying to determine the best temperature setting for a reaction to maximize yield.

- **States (S):** Different temperature settings (e.g., 20°C, 25°C, 30°C).
- **Actions (A):** Adjusting the temperature up or down.
- **Rewards (R):** The yield from the reaction at each temperature (e.g., 80% yield at 25°C, 60% yield at 20°C).

As you experiment, you would record the yield for each temperature and adjust your strategy based on the results. Over time, you learn which temperature yields the best results, similar to how an RL agent learns the best actions to take.

### 5.2.4 Environmental Types in Reinforcement Learning

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

### 5.2.5 Classifications of Reinforcement Learning

Reinforcement learning can be broadly divided into two categories:

#### 5.2.5.1 Model-Based Reinforcement Learning

In model-based reinforcement learning, the agent builds a model of the environment's dynamics. This allows the agent to simulate different actions and predict their outcomes before taking them. This approach can lead to faster learning since the agent can plan its actions based on the model.

#### 5.2.5.2 Model-Free Reinforcement Learning

Model-free reinforcement learning does not require a model of the environment. Instead, the agent learns directly from its experiences. This approach can be simpler to implement but may require more interactions with the environment to learn effectively. Popular model-free methods include Q-learning and policy gradient methods.

### 5.2.6 Understanding Value Functions

A value function estimates how good it is for the agent to be in a given state, considering the long-term reward. There are two primary types of value functions:

- **State Value Function (V):** Estimates the expected return from a state, following a specific policy.

  $$V(s) = E[R | s]$$

- **Action Value Function (Q):** Estimates the expected return from taking a specific action in a state and then following a specific policy.

  $$Q(s, a) = E[R | s, a]$$

Understanding value functions helps the agent evaluate the quality of its actions and adjust its strategy accordingly.

### 5.2.7 The Exploration-Exploitation Dilemma

One of the critical challenges in reinforcement learning is the trade-off between **exploration** and **exploitation**:

- **Exploration:** Trying new actions to discover their effects and potential rewards.
- **Exploitation:** Leveraging known actions that yield high rewards based on past experiences.

An effective RL algorithm must balance these two aspects to learn efficiently. For example, using an ε-greedy strategy, the agent occasionally selects a random action (exploration) while mostly selecting the best-known action (exploitation).

### 5.2.8 Policy-Based vs. Value-Based Approaches

Reinforcement learning can be classified into two main approaches based on how the agent learns:

- **Policy-Based Methods:** These methods directly optimize the policy, which maps states to actions. An example is the REINFORCE algorithm, which updates the policy based on the received rewards.

- **Value-Based Methods:** These methods estimate the value functions and derive the policy from them. Q-learning is a prominent example, where the agent updates its action value function based on the rewards received.

### 5.2.9 Practical Application in Chemistry

Let's implement a simple reinforcement learning scenario using Python, where we'll simulate adjusting the temperature for a chemical reaction to maximize yield. In this example, we'll use Q-learning as our learning method.

```python
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
```

This example demonstrates how a reinforcement learning agent can learn to maximize yield based on the temperature settings of a chemical reaction.

---

### Section 5.2 – Quiz Questions

#### 1) Factual Questions

##### Question 1
What is the role of the discount factor (γ) in a Markov Decision Process?

**A.** It determines the exploration probability  
**B.** It decides how often the agent updates its policy  
**C.** It balances immediate vs. future rewards  
**D.** It scales the reward function linearly

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: The discount factor γ controls how much future rewards are valued compared to immediate ones. A value closer to 1 emphasizes long-term rewards.
</details>

---

##### Question 2
Which of the following is not a component of a Markov Decision Process?

**A.** Policy Gradient  
**B.** Transition Function  
**C.** Reward Function  
**D.** States

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: A
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Policy gradient is an algorithm used in RL, but it is not a core component of the MDP, which includes states, actions, transitions, rewards, and a discount factor.
</details>

---

##### Question 3
What type of reinforcement learning does not require modeling the environment?

**A.** Model-Based RL  
**B.** Model-Free RL  
**C.** Deterministic RL  
**D.** Fully Observable RL

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Model-free RL directly learns from experiences without constructing an internal model of the environment.
</details>

---

##### Question 4
In the ε-greedy strategy, what is the purpose of the ε parameter?

**A.** It controls learning rate during backpropagation  
**B.** It adjusts the temperature in simulated annealing  
**C.** It determines how often the agent explores  
**D.** It sets the discount for future rewards

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: In ε-greedy strategies, ε represents the probability of taking a random action (exploration) instead of the best-known action (exploitation).
</details>

---

#### 2) Conceptual Questions

##### Question 5
Why might reinforcement learning be valuable for chemical experiments such as temperature optimization?

**A.** It avoids the need to run any physical experiments  
**B.** It uses genetic algorithms to mutate molecules  
**C.** It can learn optimal settings from experimental feedback  
**D.** It builds full quantum mechanical models for each compound

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Reinforcement learning iteratively improves decision-making based on rewards from previous actions, making it ideal for optimizing variables like temperature in experiments.
</details>

---

##### Question 6
What is the key difference between policy-based and value-based reinforcement learning?

**A.** Policy-based methods require a model of the environment  
**B.** Value-based methods are used only for continuous actions  
**C.** Policy-based methods optimize actions directly; value-based methods derive actions from value estimates  
**D.** Value-based methods update policies using crossover and mutation

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Policy-based methods learn the policy function directly, while value-based methods estimate value functions and derive the policy from them.
</details>

---

## 5.3 Genetic Algorithms

Genetic algorithms (GAs) are a class of evolutionary algorithms inspired by Charles Darwin's theory of natural selection. They operate by iteratively evolving a population of candidate solutions through processes that mimic biological evolution—selection, crossover (recombination), and mutation. These algorithms have found extensive applications in various scientific disciplines, particularly in molecular design and drug discovery, where they help in searching the vast chemical space for molecules with optimized properties.

Traditional brute-force methods for molecular optimization are computationally expensive due to the enormous number of possible molecular structures. In contrast, genetic algorithms provide an efficient heuristic approach by learning from previous generations and guiding the search towards more promising molecules.

In this chapter, we will explore the mechanisms, applications, advantages, limitations, and implementation of genetic algorithms in molecular property optimization.

### 5.3.1 Principles of Genetic Algorithms

A genetic algorithm follows a structured evolutionary cycle, consisting of the following main steps:

1. **Initialization:** The algorithm begins with a random population of candidate solutions (molecules), each encoded in a structured format such as SMILES strings, molecular graphs, or fingerprints.
2. **Fitness Evaluation:** Each molecule is evaluated using a fitness function that quantifies its desirability based on specific molecular properties (e.g., solubility, binding affinity).
3. **Selection:** Molecules with the best properties are more likely to be chosen as parents for reproduction.
4. **Crossover (Recombination):** Fragments from two parent molecules are combined to form offspring with new chemical structures.
5. **Mutation:** Small modifications are applied to offspring molecules, introducing diversity and allowing exploration of novel structures.
6. **Survivor Selection:** The best molecules from the population (parents and offspring) are retained for the next generation.
7. **Termination Condition:** The process is repeated until a stopping criterion is met (e.g., reaching a set number of generations or achieving an optimal fitness score).

Over successive generations, genetic algorithms refine molecular candidates toward optimal properties, effectively searching the chemical space for molecules that satisfy predefined criteria.

### 5.3.2 Applications of Genetic Algorithms in Molecular Design

Genetic algorithms have been widely used in computational chemistry and drug discovery. Some key applications include:

#### 5.3.2.1 De Novo Molecular Generation

GAs can evolve new molecular structures from scratch, optimizing for drug-likeness, bioavailability, or docking affinity with biological targets.

#### 5.3.2.2 Lead Optimization

In drug development, GAs refine existing molecular candidates by improving their efficacy, solubility, or toxicity profiles while maintaining their structural integrity.

#### 5.3.2.3 Multi-Objective Optimization

GAs can handle multiple conflicting objectives, such as maximizing biological activity while minimizing toxicity and ensuring synthetic accessibility.

#### 5.3.2.4 Reaction Optimization

GAs can help design reaction pathways to synthesize specific target molecules while optimizing for yield, cost, and environmental impact.

These applications make GAs a valuable tool for rational drug design and materials science.

### 5.3.3 Implementing Genetic Algorithms for Molecular Optimization

#### 5.3.3.1 Encoding Molecular Structures

A crucial step in applying GAs to molecular design is choosing a suitable molecular representation. Some common encodings include:

- **SMILES Representation:** Molecules are treated as strings of characters. Crossover and mutation involve modifying these strings.
- **Graph Representation:** Molecules are represented as graphs where nodes correspond to atoms and edges to chemical bonds. Mutations involve modifying nodes or edges.
- **Fingerprint-Based Representation:** Molecules are represented as binary feature vectors encoding molecular properties (e.g., MACCS, Morgan fingerprints).

The choice of representation impacts how genetic operations are applied.

#### 5.3.3.2 Defining the Fitness Function

The fitness function evaluates each molecule's desirability based on specific molecular properties. Some commonly used objectives include:
- **Lipophilicity (LogP):** Determines how well a drug dissolves in fats versus water.
- **Molecular Weight (MW):** Drugs must have an appropriate MW for absorption and distribution in the body.
- **Synthetic Feasibility:** Ensures that generated molecules can be synthesized in a lab.
- **Drug-Likeness:** Uses metrics such as Lipinski's Rule of Five to assess whether a molecule is suitable as a drug candidate.

The fitness function is crucial because it guides the evolutionary process toward the desired molecular properties.

#### 5.3.3.3 Selection, Crossover, and Mutation Strategies

Once the molecules are evaluated, the GA applies selection, crossover, and mutation to generate the next generation of molecules.

**Selection Strategies**
- **Roulette Wheel Selection:** Molecules are chosen probabilistically based on their fitness.
- **Tournament Selection:** A subset of molecules competes, and the best are selected.

**Crossover Strategies**
- **One-Point Crossover:** A single cut is made in the molecule, and fragments from two parents are swapped.
- **Two-Point Crossover:** Two cut points are used for recombination.
- **Graph-Based Crossover:** Molecular substructures are exchanged between parents.

**Mutation Strategies**
- **SMILES-Based Mutation:** Randomly changing characters in the SMILES string.
- **Graph-Based Mutation:** Randomly adding or deleting atoms/bonds.
- **Fingerprint Mutation:** Flipping bits in the molecular fingerprint vector.

Tuning these genetic operators balances exploration (diversity) and exploitation (optimization).

#### 5.3.3.4 Genetic Algorithms Example

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/18EQfIEUt72nzruy4_2rb0aAXwglOT5C_?usp=sharing)

**Step 1: Install and Import Required Dependencies**
```python
# Install necessary libraries in Google Colab
!pip install rdkit-pypi deap
import random
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from deap import base, creator, tools, algorithms
import pandas as pd
```

**Step 2: Load and Process the Dataset**

```python
# Load the ESOL dataset (water solubility dataset)
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
data = pd.read_csv(url)

# Select SMILES and solubility columns
data = data[['smiles', 'measured log solubility in mols per litre']]
data.columns = ['SMILES', 'Solubility']

# Filter out invalid SMILES before starting
valid_smiles_list = [s for s in data["SMILES"].tolist() if Chem.MolFromSmiles(s)]
if not valid_smiles_list:
    raise ValueError("No valid SMILES found in the dataset.")

print(f"Loaded {len(valid_smiles_list)} valid molecules.")
```

**Step 3: Define the Genetic Algorithm Compounds**

```python
def fitness_function(individual):
    """
    Evaluates the fitness of a molecule based on LogP.
    Higher LogP means higher lipophilicity.
    """
    if isinstance(individual, list) and len(individual) > 0:
        smiles = individual[0]  # Extract SMILES string
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            logP = Descriptors.MolLogP(mol)
            return logP,  # Fitness must be returned as a tuple
    return -1000,  # Assign very low fitness for invalid molecules
        
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize LogP
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_smiles", lambda: random.choice(valid_smiles_list))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_smiles, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def crossover_smiles(ind1, ind2):
    """
    Performs crossover between two SMILES strings by swapping halves.
    """
    if isinstance(ind1, list) and isinstance(ind2, list) and len(ind1[0]) > 1 and len(ind2[0]) > 1:
        cut = random.randint(1, min(len(ind1[0]), len(ind2[0])) - 1)
        ind1[0], ind2[0] = ind1[0][:cut] + ind2[0][cut:], ind2[0][:cut] + ind1[0][cut:]
    return ind1, ind2

def mutate_smiles(individual, max_attempts=10):
    """
    Attempts to mutate the SMILES string while ensuring validity.
    Retries up to max_attempts times if the mutation creates an invalid molecule.
    """
    for _ in range(max_attempts):
        if isinstance(individual, list) and len(individual[0]) > 0:
            idx = random.randint(0, len(individual[0]) - 1)
            new_char = random.choice("CHON")  # Common organic elements
            mutated_smiles = individual[0][:idx] + new_char + individual[0][idx+1:]

            if Chem.MolFromSmiles(mutated_smiles):  # Ensure valid mutation
                individual[0] = mutated_smiles
                return individual,

    return individual,  # Return the original molecule if no valid mutation found

toolbox.register("mate", crossover_smiles)
toolbox.register("mutate", mutate_smiles)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)
```

**Step 4: Run the Genetic Algorithm**

```python
# Genetic Algorithm Parameters
POP_SIZE = 100
GENS = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

# Initialize population
population = toolbox.population(n=POP_SIZE)

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE, ngen=GENS, verbose=True)
```

**Step 5: Select and Validate the Best Molecule**
```python
# Filter out invalid molecules before selecting the best one
valid_population = [ind for ind in population if Chem.MolFromSmiles(ind[0])]
if not valid_population:
    print("No valid molecules were generated. Re-initializing population.")
    population = toolbox.population(n=POP_SIZE)
    valid_population = [ind for ind in population if Chem.MolFromSmiles(ind[0])]


best_individual = tools.selBest(valid_population, k=1)[0]
print("Best Valid Molecule:", best_individual[0], "LogP:", fitness_function(best_individual)[0])
```

**Step 6: Visualize the Best Model**
```python
def visualize_molecule(smiles):
    """
    Converts a SMILES string to an RDKit molecular image.
    If the SMILES is invalid, it returns an error message.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    return Draw.MolToImage(mol, size=(200, 200))

if best_individual:
    best_smiles = best_individual[0]
    image = visualize_molecule(best_smiles)
    if image:
        display(image)
    else:
        print("Could not generate a valid molecule for visualization.")
```

**Step 7: Track Optimization Progress**
```python
# Ensure only valid fitness scores are plotted
fitness_scores = []
for ind in population:
    if isinstance(ind, list) and len(ind) > 0:
        smiles = ind[0]  # Extract SMILES
        if Chem.MolFromSmiles(smiles):  # Validate molecule
            fitness_scores.append(ind.fitness.values[0])

if fitness_scores:
    plt.plot(range(len(fitness_scores)), fitness_scores, marker='o', linestyle='-', color='b')
    plt.xlabel("Generation")
    plt.ylabel("Best LogP Score")
    plt.title("Genetic Algorithm Optimization Progress")
    plt.grid()
    plt.show()
else:
    print("No valid fitness scores to plot.")
```

---

### Section 5.3 – Quiz Questions

#### 1) Factual Questions

##### Question 1
What is the purpose of the fitness function in a genetic algorithm applied to molecular design?

**A.** It encodes molecular structures into SMILES format  
**B.** It initializes the population of molecules  
**C.** It quantifies how desirable a molecule is based on defined properties  
**D.** It randomly mutates molecules to increase diversity

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: The fitness function evaluates how well a molecule satisfies desired objectives such as solubility, LogP, or drug-likeness, guiding the selection process.
</details>

---

##### Question 2
Which of the following is not a typical genetic operator in molecular optimization?

**A.** Crossover  
**B.** Mutation  
**C.** Evaluation  
**D.** Aggregation

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: D
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Aggregation is not a standard genetic operator. Genetic algorithms typically use crossover, mutation, selection, and evaluation.
</details>

---

##### Question 3
Which representation method encodes a molecule as a string of atoms and bonds?

**A.** Graph representation  
**B.** Fingerprint-based representation  
**C.** SMILES representation  
**D.** Adjacency matrix representation

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: SMILES (Simplified Molecular Input Line Entry System) encodes a molecule as a text string representing its atoms and bonds.
</details>

---

##### Question 4
What is the main function of mutation in genetic algorithms?

**A.** To copy molecules from one generation to the next  
**B.** To combine fragments from two parents  
**C.** To introduce structural diversity into the population  
**D.** To sort molecules by LogP

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Mutation introduces small, random changes to molecules, helping explore new areas of chemical space and prevent premature convergence.
</details>

---

#### 2) Conceptual Questions

##### Question 5
Why are genetic algorithms especially useful in molecular optimization compared to brute-force methods?

**A.** They always find the globally optimal molecule in one generation  
**B.** They use deterministic rules to eliminate randomness  
**C.** They explore chemical space efficiently through evolution-inspired heuristics  
**D.** They avoid the need for molecular structure encodings

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: GAs search large and complex chemical spaces by evolving better solutions over generations, making them far more efficient than brute-force enumeration.
</details>

---

##### Question 6
What is the role of selection strategies like roulette wheel or tournament selection in GAs?

**A.** They apply mutations to individual atoms  
**B.** They visualize molecules during optimization  
**C.** They choose high-performing molecules for reproduction  
**D.** They encode molecules into binary fingerprints

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Selection strategies determine which molecules are chosen to reproduce, often favoring those with higher fitness scores to guide the algorithm toward better candidates.
</details>

---

## 5.4 Generative models with conditions

Generative models are a class of machine learning models designed to create new data samples that resemble a given dataset. Unlike traditional models used for classification or regression, generative models aim to model the underlying data distribution, enabling them to generate realistic, novel examples. This makes them highly valuable for applications such as image synthesis, natural language generation, and molecule design.

In molecular property optimization, generative models play a crucial role by generating novel molecular structures that meet specific criteria, such as high binding affinity or low toxicity. These models can be conditioned on desired molecular properties, allowing efficient exploration of chemical space.

Conditioning generative models involves incorporating property-specific information into the model to ensure generated molecules meet predefined criteria. For example, in variational autoencoders (VAEs), the desired property values are added as inputs to the encoder and decoder networks, facilitating the generation of property-conditioned molecules. Similarly, generative adversarial networks (GANs) can incorporate property constraints either in the generator input or through a modified loss function.

Reinforcement learning (RL) is also commonly used with generative models. Here, an RL agent evaluates candidate molecules proposed by the generative model, guiding the process toward optimal molecular regions based on predicted properties. This approach has been effective in optimizing characteristics like drug-likeness and bioactivity.

Sequence-based models such as conditional recurrent neural networks (RNNs) and transformer architectures have further enhanced molecular generation by capturing complex structural dependencies. These models generate diverse molecular structures tailored to specific design criteria when conditioned on relevant features.

In summary, generative models with conditions offer a powerful framework for accelerating molecular discovery. By enabling the targeted design of novel molecules, they reduce experimental costs and open new avenues in drug development and materials science.

### 5.4.1 What are Conditional Generative Models?

Conditional generative models extend the concept of generative modeling by incorporating additional information—known as conditions—to guide the generation process. These conditions act as constraints or directives that steer the model toward creating samples with specific characteristics or properties. For instance, while a standard generative model for molecules might generate chemically valid compounds randomly, a conditional generative model can be directed to generate molecules with high solubility, low toxicity, or a specific structural feature.

The addition of conditions allows for greater control and precision in the generative process. This is particularly valuable in applications like molecular property optimization, where the goal is not just to create valid molecules but to ensure they meet predefined requirements for properties like binding affinity, pharmacokinetics, or synthetic accessibility.

### 5.4.2 Why are Generative Models with Conditions Important

In the field of molecular design, the chemical space—comprising all theoretically possible molecules—is vast and largely unexplored. Traditional trial-and-error methods of exploring this space are slow, resource-intensive, and often limited in their ability to optimize multiple properties simultaneously. Generative models, and specifically conditional generative models, address this challenge by:

- **Guided Exploration:** Allowing researchers to specify desired molecular properties, thus narrowing the search space.
- **Accelerated Discovery:** Rapidly generating candidate molecules with high probability of success in real-world applications.
- **Multi-Objective Optimization:** Balancing trade-offs between competing molecular properties, such as efficacy and safety.

For example, in drug discovery, a conditional generative model might be tasked with generating molecules that are not only effective against a particular target protein but are also non-toxic and metabolically stable. Similarly, in materials science, conditional models can help design polymers with specific mechanical or thermal properties.

**Applications Across Domains**

Conditional generative models have proven their utility across various domains:
- **Drug Discovery:** Generating molecules with tailored ADMET (absorption, distribution, metabolism, excretion, and toxicity) profiles.
- **Materials Science:** Designing compounds with specific physical or chemical properties, such as conductivity or strength.
- **Synthetic Chemistry:** Optimizing molecules for ease of synthesis or compatibility with specific reaction pathways.

### 5.4.3 Incorporating Conditions in Generative Models

To generate data that satisfies specific requirements, generative models need a mechanism to include the desired conditions as part of the generation process. This is achieved in several ways, depending on the model architecture:

**Input Embedding and Concatenation**

- Conditions are treated as additional inputs, often concatenated with the primary data representation. For example:
  - In a molecular model using SMILES (Simplified Molecular Input Line Entry System) strings, conditions such as desired solubility can be concatenated as numerical features.
  - In graph-based models, properties like binding affinity may be embedded alongside the graph node and edge features.
- This method works well with architectures like neural networks, where inputs can be treated as feature vectors.

**Latent Space Conditioning**

- In models like Variational Autoencoders (VAEs), conditions are embedded into the latent space—a compressed representation of the data.
- By augmenting the latent space with property-specific information, the model learns to decode latent vectors into outputs that reflect both the data distribution and the specified conditions.
- Example: A VAE trained on molecules can be conditioned to generate molecules with a specific melting point by embedding the desired melting point in the latent vector.

**Adversarial Conditioning**
- Generative Adversarial Networks (GANs) achieve conditioning by incorporating the desired properties into both the generator and discriminator networks.
- The generator produces samples that aim to satisfy the conditions, while the discriminator evaluates both the sample validity and whether the conditions are met.
- Example: In a cGAN (Conditional GAN), the generator might create molecules with a given molecular weight, and the discriminator ensures that the generated molecules align with this condition.

**Attention-Based Conditioning**
- Attention mechanisms, commonly used in transformer models, can focus on specific parts of the input data or property representations, allowing fine-grained control over the generated output.
- Example: In a transformer trained on molecular graphs, the attention mechanism can emphasize functional groups that align with a desired chemical property.

### 5.4.4 Types of Conditional Generative Models

Several types of generative models have been adapted for conditional use in molecular property optimization. Each has distinct strengths and weaknesses, depending on the complexity of the task and the nature of the conditions:
- cVAEs are computationally efficient and ideal for tasks requiring smooth latent space interpolation.
- cGANs excel at generating high-quality and diverse outputs.
- Transformers are powerful for handling complex molecular representations and large datasets.
- Reinforcement learning approaches are well-suited for optimizing challenging or non-differentiable objectives.

By leveraging these models, researchers can efficiently explore chemical space while ensuring that generated molecules meet specified requirements. The section below illustrates specific breakdowns for each type of model.

**Conditional Variational Autoencoders (cVAEs)**
- **Architecture:**
  - A cVAE consists of an encoder-decoder framework where conditions are introduced into the latent space.
- **How It Works:**
  - The encoder compresses molecular representations into a latent vector while embedding the conditions.
  - The decoder generates new molecules from the latent vector, ensuring they align with the conditions.
- **Application Example:**
  - Generating drug-like molecules with high solubility and low toxicity.

**Conditional Generative Adversarial Networks (cGANs)**
- **Architecture:**
  - A cGAN consists of a generator-discriminator pair, with conditions embedded into both networks.
- **How It Works:**
  - The generator creates samples aimed at satisfying the conditions, and the discriminator evaluates their validity and adherence to the specified properties.
- **Application Example:**
  - Designing polymers with specific thermal or mechanical properties.

**Transformer-Based Conditional Models**
- **Architecture:**
  - Transformer models use self-attention mechanisms to handle sequence or graph data.
- **How It Works:**
  - Conditions are integrated as input embeddings or used to modulate attention weights, allowing precise control over generated structures.
- **Application Example:**
  - Generating molecular graphs with desired binding affinity to a target protein.

**Reinforcement Learning-Augmented Generative Models**
- **Architecture:**
  - Generative models are combined with reinforcement learning to optimize for conditions during the generation process.
- **How It Works:**
  - A reward function evaluates the generated molecules for property adherence, and the model updates its parameters to maximize this reward.
- **Application Example:**
  - Designing molecules with high binding affinity and low synthetic complexity.

### 5.4.5 Representations and Conditioning Strategies

Effective molecular property optimization using conditional generative models depends heavily on the choice of molecular representation and the encoding of property conditions. This section explores the key strategies for representing molecules and encoding conditions, along with the trade-offs between representation complexity and model performance.

#### 5.4.5.1 Representing Molecules for Generative Models

Different molecular representations influence the performance and efficiency of generative models. The most commonly used representations are:

- **SMILES Strings:**
  - A linear textual representation of molecular structures.
  - Simple to preprocess and widely used for sequence-based models like RNNs, VAEs, and transformers.
  - **Advantages:**
    - Easy to work with using standard NLP techniques.
    - Compact and human-readable.
  - **Challenges:**
    - Sensitive to small changes (e.g., one character change can drastically alter the molecule).
    - Does not explicitly capture 3D structural information.

- **Molecular Graphs:**
  - Represent molecules as graphs, with atoms as nodes and bonds as edges.
  - Suitable for graph neural networks (GNNs) or graph-based generative models.
  - **Advantages:**
    - Captures the underlying structure and relationships between atoms.
    - More robust for tasks involving structural optimization.
  - **Challenges:**
    - Higher computational complexity compared to SMILES.
    - Requires specialized graph-based architectures.

- **3D Structures:**
  - Include spatial coordinates of atoms, representing molecular conformations in 3D space.
  - Used in models that integrate quantum-chemical calculations or focus on protein-ligand interactions.
  - **Advantages:**
    - Provides detailed information about molecular geometry and interactions.
    - Crucial for applications in drug binding and docking studies.
  - **Challenges:**
    - Computationally intensive to handle and process.
    - Requires additional data (e.g., X-ray crystallography or simulations).

#### 5.4.5.2 Encoding Conditions

The conditions used to guide generative models can be scalar (numerical) or categorical, depending on the property to be optimized.
- **Scalar Properties:**
  - Examples: Solubility, binding affinity, molecular weight.
  - **Encoding:**
    - Directly use numerical values or normalize them within a specific range.
    - Feed as continuous input to the model or integrate into the latent space.
  - **Application:** Generate molecules with a specific solubility threshold or within a molecular weight range.

- **Categorical Data:**
  - Examples: Functional groups, molecule types, or chemical classes.
  - **Encoding:**
    - Use one-hot encoding, where each category is represented as a binary vector.
    - Alternatively, embed categorical data as dense vectors using an embedding layer.
  - **Application:** Generate molecules that belong to a specific class (e.g., benzene derivatives) or contain certain functional groups (e.g., hydroxyl groups).

#### 5.4.5.3 Trade-Offs Between Representation Complexity and Model Performance
- **Simpler Representations (e.g., SMILES):**
  - Easier to preprocess and faster to train.
  - Risk of losing critical structural or spatial information.
- **Complex Representations (e.g., molecular graphs, 3D structures):**
  - Provide richer information about the molecule.
  - Require more computational resources and specialized architectures.

The choice of representation and encoding depends on the task's requirements, the computational budget, and the complexity of the target properties.

### 5.4.6 Applications in Molecular Property Optimization

Conditional generative models have demonstrated transformative potential across various domains in molecular property optimization. This section highlights key applications, emphasizing their role in accelerating discovery and improving the efficiency of molecular design.

#### 5.4.6.1 Drug Discovery

- **Generating Molecules with Specific ADMET Properties:**
  - ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties are crucial in drug development. Conditional generative models can produce molecules that satisfy these requirements, ensuring better pharmacokinetics and reduced side effects.
  - Example: A cVAE can be trained to generate molecules with high bioavailability and low toxicity.
- **Ligand-Based and Structure-Based Drug Design:**
  - Ligand-based models generate molecules based on known active compounds, optimizing specific properties like binding affinity.
  - Structure-based models use 3D information about the target protein to generate molecules that fit its binding pocket.
  - Example: A conditional GAN could generate ligands with optimal binding to a specific enzyme, improving the lead optimization process.

#### 5.4.6.2 Materials Science

- **Designing Polymers with Target Mechanical Properties:**
  - Polymers with specific mechanical properties, such as high elasticity, strength, or thermal stability, are crucial in materials design.
  - Conditional models allow the generation of polymer candidates that meet predefined criteria, reducing the time required for experimental testing.
  - Example: A transformer-based model can generate polymer structures optimized for tensile strength while maintaining low production costs.
- **Thermal Properties Optimization:**
  - Generate materials with specific thermal conductivity or resistance to heat, which are essential in electronics or aerospace industries.
  - Example: Use a graph-based conditional model to design materials for heat sinks with high thermal conductivity.

#### 5.4.6.3 Synthetic Chemistry

- **Optimizing Molecules for Synthesis Feasibility:**
  - Molecules designed through generative models can sometimes be difficult or expensive to synthesize. Conditional models can optimize for synthetic accessibility, ensuring that generated molecules can be produced using available chemical pathways.
  - Example: A cGAN could generate molecules that minimize the number of synthetic steps while maintaining desired properties like high yield.
- **Designing Molecules for Specific Reaction Pathways:**
  - Conditional models can generate molecules that align with specific reaction mechanisms or catalytic conditions, aiding in reaction design and optimization.
  - Example: Generate precursors for a specific polymerization reaction to produce biodegradable plastics.

### 5.4.7 Generative Models with Conditions Full Manual Process

**Completed and Compiled Code:** [Click Here](https://colab.research.google.com/drive/18EQfIEUt72nzruy4_2rb0aAXwglOT5C_?usp=sharing)

**Step 1: Install Required Libraries**
```python
!pip install rdkit-pypi
!pip install tensorflow
!pip install numpy pandas
import tensorflow as tf
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.model_selection import train_test_split
```

**Step 2: Download and Upload the ESOL Dataset from Deepchemdata**

The **ESOL Dataset** is available on Github: [Click Here](https://github.com/deepchem/deepchem/blob/master/datasets/delaney-processed.csv)

```python
# Load the ESOL dataset (direct source: https://github.com/deepchem/deepchem/blob/master/datasets/delaney-processed.csv)
url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
data = pd.read_csv(url)

# Display the first few rows of the dataset
data.head()
```

**Step 3: Data Preprocessing and Encoding**

```python
# Tokenize SMILES strings
data['tokenized_smiles'] = data['smiles'].apply(lambda x: list(x))

# Create a vocabulary of unique characters
vocab = sorted(set(''.join(data['smiles'].values)))
vocab.append(' ')  # Add a padding character
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Encode SMILES strings
max_length = max(data['tokenized_smiles'].apply(len))
data['encoded_smiles'] = data['tokenized_smiles'].apply(
    lambda x: [char_to_idx[char] for char in x] + [char_to_idx[' ']] * (max_length - len(x))
)

# Normalize solubility values
data['normalized_solubility'] = (data['measured log solubility in mols per litre'] - data['measured log solubility in mols per litre'].mean()) / data['measured log solubility in mols per litre'].std()

# Prepare input arrays
X = np.array(data['encoded_smiles'].tolist())
y = data['normalized_solubility'].values.reshape(-1, 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Step 4: Conditional Variational Autoencoder (cVAE) Architecture**

```python
# Set parameters
latent_dim = 50  # Dimensionality of the latent space
input_dim = X_train.shape[1]  # Length of the input sequences
vocab_size = len(vocab)  # Size of the vocabulary

# Encoder
smiles_input = layers.Input(shape=(input_dim,), name='SMILES_Input')
condition_input = layers.Input(shape=(1,), name='Condition_Input')

# Embed the SMILES input
x = layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=input_dim)(smiles_input)
x = layers.Concatenate()([x, layers.RepeatVector(input_dim)(condition_input)])
x = layers.LSTM(128, return_sequences=False)(x)

# Latent space
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,), name='Latent_Input')
x = layers.Concatenate()([latent_inputs, condition_input])
x = layers.RepeatVector(input_dim)(x)
x = layers.LSTM(128, return_sequences=True)(x)
decoded = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(x)

# Define models
encoder = models.Model([smiles_input, condition_input], [z_mean, z_log_var, z], name='Encoder')
decoder = models.Model([latent_inputs, condition_input], decoded, name='Decoder')

# cVAE model
outputs = decoder([encoder([smiles_input, condition_input])[2], condition_input])
cvae = models.Model([smiles_input, condition_input], outputs, name='cVAE')
```

**Step 5: Custom cVAE Model Implementation and Training**
```python
# Custom VAE class with integrated loss
class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        smiles_input, condition_input = inputs
        z_mean, z_log_var, z = self.encoder([smiles_input, condition_input])
        reconstructed = self.decoder([z, condition_input])

        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.sparse_categorical_crossentropy(smiles_input, reconstructed)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        kl_loss = tf.reduce_mean(kl_loss)

        # Add the total loss
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)

        return reconstructed

# Compile the cVAE model
cvae = CVAE(encoder, decoder)
cvae.compile(optimizer="adam")

# Train the model
cvae.fit([X_train, y_train], epochs=50, batch_size=32, validation_data=([X_test, y_test], None))

# code may take a bit of time to run as we are training 50 epochs
```

**Step 6: Molecule Generation, Evaluation, and Visualization**
```python
# Improved decoding using probability sampling instead of argmax
def decode_smiles_probabilistic(encoded_output, idx_to_char):
    smiles = ''
    for timestep in encoded_output:
        # Sample from the probability distribution to introduce diversity
        sampled_index = np.random.choice(len(timestep), p=timestep / np.sum(timestep))
        char = idx_to_char[sampled_index]
        if char != ' ':  # Skip padding
            smiles += char
    return smiles

# Generate diverse molecules based on desired solubility
def generate_molecules(desired_solubility, num_samples=10):
    generated_smiles_list = []
    
    for _ in range(num_samples):
        # Sample a random point from the latent space (standard normal distribution)
        latent_sample = np.random.normal(size=(1, 50))  # 50 is the latent_dim
        condition = np.array([[desired_solubility]])    # Desired solubility condition
        
        # Generate SMILES using the decoder
        generated_output = decoder.predict([latent_sample, condition])
        
        # Decode using probabilistic sampling
        generated_smiles = decode_smiles_probabilistic(generated_output[0], idx_to_char)
        generated_smiles_list.append(generated_smiles)
    
    return generated_smiles_list

# Evaluate and validate the generated molecules
def evaluate_molecules(smiles_list):
    valid_molecules = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:  # Check for chemical validity
                mol_weight = Descriptors.MolWt(mol)
                logP = Descriptors.MolLogP(mol)
                print(f"✅ Valid SMILES: {smiles} | MolWt: {mol_weight:.2f}, LogP: {logP:.2f}")
                valid_molecules.append(smiles)
            else:
                print(f"Invalid SMILES: {smiles}")
        except:
            print(f"Error processing SMILES: {smiles}")
    return valid_molecules

# Optional: Visualize valid molecules
def visualize_molecules(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
    return Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))

# Example: Generate 10 molecules with high solubility (normalized around 0.9)
generated_molecules = generate_molecules(desired_solubility=0.9, num_samples=10)

# Display generated molecules
print("\nGenerated Molecules:")
for idx, smiles in enumerate(generated_molecules):
    print(f"Molecule {idx + 1}: {smiles}")

# Evaluate generated molecules for chemical validity
valid_molecules = evaluate_molecules(generated_molecules)

# Visualize valid molecules
if valid_molecules:
    display(visualize_molecules(valid_molecules))
else:
    print("No valid molecules generated.")
```

**Step 7: Latent Space Visualization Using PCA**
```python
# Step 1: Inverse PCA to get the original latent vector
# Ensure 'pca' is the same PCA object used for dimensionality reduction
high_solubility_latent_2d = np.array([[ -0.01, 0.006 ]])  # Example coordinates from PCA plot
high_solubility_latent = pca.inverse_transform(high_solubility_latent_2d)  # Convert back to original latent space (50 dimensions)

# Step 2: Generate molecule
condition = np.array([[0.9]])  # High solubility condition

# Step 3: Decode
generated_output = decoder.predict([high_solubility_latent.reshape(1, -1), condition])
generated_smiles = decode_smiles_probabilistic(generated_output[0], idx_to_char)

# Step 4: Display the generated molecule
print(f"Generated molecule for high solubility region: {generated_smiles}")

# Generate molecules from both ends of the solubility spectrum
solubility_points = [(-0.01, 0.006), (0.015, 0.002)]  # Example: High and low solubility regions

for coords in solubility_points:
    latent_vector = pca.inverse_transform(np.array([coords]))
    condition = np.array([[0.9]])  # Adjust solubility condition as needed
    
    generated_output = decoder.predict([latent_vector.reshape(1, -1), condition])
    generated_smiles = decode_smiles_probabilistic(generated_output[0], idx_to_char)
    
    print(f"Generated molecule from latent space coordinates {coords}: {generated_smiles}")
```

**Step 8: Latent Space Exploration and Molecule Generation Based on Solubility**
```python
# Step 1: Inverse PCA to get the original latent vector
# Ensure 'pca' is the same PCA object used for dimensionality reduction
high_solubility_latent_2d = np.array([[ -0.01, 0.006 ]])  # Example coordinates from PCA plot
high_solubility_latent = pca.inverse_transform(high_solubility_latent_2d)  # Convert back to original latent space (50 dimensions)

# Step 2: Generate molecule
condition = np.array([[0.9]])  # High solubility condition

# Step 3: Decode
generated_output = decoder.predict([high_solubility_latent.reshape(1, -1), condition])
generated_smiles = decode_smiles_probabilistic(generated_output[0], idx_to_char)

# Step 4: Display the generated molecule
print(f"Generated molecule for high solubility region: {generated_smiles}")

# Generate molecules from both ends of the solubility spectrum
solubility_points = [(-0.01, 0.006), (0.015, 0.002)]  # Example: High and low solubility regions

for coords in solubility_points:
    latent_vector = pca.inverse_transform(np.array([coords]))
    condition = np.array([[0.9]])  # Adjust solubility condition as needed
    
    generated_output = decoder.predict([latent_vector.reshape(1, -1), condition])
    generated_smiles = decode_smiles_probabilistic(generated_output[0], idx_to_char)
    
    print(f"Generated molecule from latent space coordinates {coords}: {generated_smiles}")
```

**Step 9: Latent Space Targeting for Property-Driven Molecule Generation and Visualization**
```python
# Step 1: Encode the test data to obtain the latent representations
z_mean, _, _ = encoder.predict([X_test, y_test])  # Encoder outputs for PCA
pca = PCA(n_components=2)                         # PCA for dimensionality reduction
z_pca = pca.fit_transform(z_mean)                 # Apply PCA

# Step 2: Define points of interest (for high and low solubility regions)
solubility_points = [(-0.01, 0.006), (0.015, 0.002)]  # Example points from the PCA plot
labels = ["High Solubility", "Low Solubility"]

# Step 3: Visualize the latent space with solubility gradient
plt.figure(figsize=(10, 6))
scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=y_test.flatten(), cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Normalized Solubility')

# Pinpoint the selected regions
for (x, y), label in zip(solubility_points, labels):
    plt.scatter(x, y, color='red', marker='X', s=100, label=label)
    plt.text(x + 0.0005, y, label, fontsize=9, verticalalignment='bottom')

plt.title('Latent Space Visualization with Target Points')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.show()

# Step 4: Generate molecules from selected points
def decode_smiles_probabilistic(encoded_output, idx_to_char):
    smiles = ''
    for timestep in encoded_output:
        sampled_index = np.random.choice(len(timestep), p=timestep / np.sum(timestep))
        char = idx_to_char[sampled_index]
        if char != ' ':
            smiles += char
    return smiles

def generate_molecules_from_latent(pca_coords, condition_value=0.9):
    latent_vector = pca.inverse_transform(np.array([pca_coords]))
    condition = np.array([[condition_value]])  # Desired solubility condition
    generated_output = decoder.predict([latent_vector.reshape(1, -1), condition])
    generated_smiles = decode_smiles_probabilistic(generated_output[0], idx_to_char)
    return generated_smiles

# Step 5: Generate and visualize molecules
generated_molecules = []
for coords, label in zip(solubility_points, labels):
    smiles = generate_molecules_from_latent(coords, condition_value=0.9)
    generated_molecules.append((label, smiles))
    print(f"{label} Region → Generated SMILES: {smiles}")

# Step 6: Visualize generated molecules
def visualize_generated_molecules(smiles_list):
    mols = []
    for _, smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol if mol else Chem.MolFromSmiles('C'))  # Placeholder for invalid SMILES
    return Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(200, 200))

# Display generated molecules
visualize_generated_molecules(generated_molecules)
```

---

### Section 5.4 – Quiz Questions

#### 1) Factual Questions

##### Question 1
What is the main purpose of a conditional generative model in molecular property optimization?

**A.** To randomly generate chemically valid molecules  
**B.** To filter invalid molecules after generation  
**C.** To generate molecules that meet specific predefined property criteria  
**D.** To simulate chemical reactions in real time

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Conditional generative models incorporate desired property constraints (like solubility or toxicity) to steer generation toward molecules that meet specific goals.
</details>

---

##### Question 2
Which of the following is a valid method for incorporating conditions into a generative model?

**A.** Deleting features from the input  
**B.** Concatenating property values with model inputs or latent representations  
**C.** Training without property labels  
**D.** Converting SMILES into image format

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Conditions can be concatenated with input sequences or latent vectors to guide generation toward molecules with desired properties.
</details>

---

##### Question 3
Which model architecture uses attention mechanisms for conditional molecular generation?

**A.** Conditional GAN  
**B.** Conditional VAE  
**C.** Transformer-based model  
**D.** Graph-based classifier

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Transformer-based models use self-attention to process input data, including conditioning vectors, allowing for complex control over molecular generation.
</details>

---

##### Question 4
Which molecular representation is most suitable for 3D property-aware generation?

**A.** SMILES string  
**B.** One-hot encoding  
**C.** Graph representation  
**D.** 3D atomic coordinates

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: D
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: 3D atomic coordinates are needed when modeling spatial interactions and geometry-dependent properties like docking or quantum behavior.
</details>

---

#### 2) Conceptual Questions

##### Question 5
Why are conditional generative models important in drug discovery?

**A.** They automate the synthesis of molecules  
**B.** They evaluate the toxicity of known drugs  
**C.** They can generate molecules that satisfy multiple property constraints simultaneously  
**D.** They improve quantum chemical accuracy

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: Conditional models allow for multi-objective optimization (e.g., high efficacy and low toxicity), which is crucial in drug discovery.
</details>

---

##### Question 6
What is the trade-off between using SMILES strings and molecular graphs in generative models?

**A.** SMILES strings require less GPU memory but cannot represent large molecules  
**B.** Molecular graphs are easier to train but lack structural information  
**C.** SMILES are simpler and faster but less robust to small changes than graphs  
**D.** Graphs are compact but harder to interpret than SMILES

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation: SMILES are simple to process with NLP models but are sensitive to small token changes. Graphs capture structural relationships more accurately but are computationally heavier.
</details>
