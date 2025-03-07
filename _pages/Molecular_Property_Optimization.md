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

The EDBOplus library presents an easy shortcut option without having to understand the full process of Bayesian Optimization: [https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/tutorials/1_CLI_example.ipynb](https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/tutorials/1_CLI_example.ipynb)

### 5.1.4 Bayesian Optimization for Molecular Property Optimization Full Manual Process

In the event where we may want to do something more complex, here is the full process laid out in an example looking to optimize logP values:

google colab: [https://colab.research.google.com/drive/1uXjcu_bzygtia9xgEHN76xMmQtcV0sY-?usp=sharing](https://colab.research.google.com/drive/1uXjcu_bzygtia9xgEHN76xMmQtcV0sY-?usp=sharing)

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
Genetic algorithms (GAs) are a class of evolutionary algorithms inspired by Charles Darwin’s theory of natural selection. They operate by iteratively evolving a population of candidate solutions through processes that mimic biological evolution—selection, crossover (recombination), and mutation. These algorithms have found extensive applications in various scientific disciplines, particularly in molecular design and drug discovery, where they help in searching the vast chemical space for molecules with optimized properties.

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

The fitness function evaluates each molecule’s desirability based on specific molecular properties. Some commonly used objectives include:
- **Lipophilicity (LogP):** Determines how well a drug dissolves in fats versus water.
- **Molecular Weight (MW):** Drugs must have an appropriate MW for absorption and distribution in the body.
- **Synthetic Feasibility:** Ensures that generated molecules can be synthesized in a lab.
- **Drug-Likeness:** Uses metrics such as Lipinski’s Rule of Five to assess whether a molecule is suitable as a drug candidate.

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


- Guided Exploration: Allowing researchers to specify desired molecular properties, thus narrowing the search space.
- Accelerated Discovery: Rapidly generating candidate molecules with high probability of success in real-world applications.
- Multi-Objective Optimization: Balancing trade-offs between competing molecular properties, such as efficacy and safety.

For example, in drug discovery, a conditional generative model might be tasked with generating molecules that are not only effective against a particular target protein but are also non-toxic and metabolically stable. Similarly, in materials science, conditional models can help design polymers with specific mechanical or thermal properties.

**Applications Across Domains**


- Conditional generative models have proven their utility across various domains:
- Drug Discovery: Generating molecules with tailored ADMET (absorption, distribution, metabolism, excretion, and toxicity) profiles.
- Materials Science: Designing compounds with specific physical or chemical properties, such as conductivity or strength.
- Synthetic Chemistry: Optimizing molecules for ease of synthesis or compatibility with specific reaction pathways.
  
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
- Architecture:
    - A cVAE consists of an encoder-decoder framework where conditions are introduced into the latent space.
- How It Works:
    - The encoder compresses molecular representations into a latent vector while embedding the conditions.
    - The decoder generates new molecules from the latent vector, ensuring they align with the conditions.
- Application Example:
    - Generating drug-like molecules with high solubility and low toxicity.
      
**Conditional Generative Adversarial Networks (cGANs)**
- Architecture:
    - A cGAN consists of a generator-discriminator pair, with conditions embedded into both networks.
- How It Works:
    - The generator creates samples aimed at satisfying the conditions, and the discriminator evaluates their validity and adherence to the specified properties.
- Application Example:
    - Designing polymers with specific thermal or mechanical properties.

**Transformer-Based Conditional Models**
- Architecture:
    - Transformer models use self-attention mechanisms to handle sequence or graph data.
- How It Works:
    - Conditions are integrated as input embeddings or used to modulate attention weights, allowing precise control over generated structures.
- Application Example:
    - Generating molecular graphs with desired binding affinity to a target protein.

**Reinforcement Learning-Augmented Generative Models**
- Architecture:
    - Generative models are combined with reinforcement learning to optimize for conditions during the generation process.
- How It Works:
    - A reward function evaluates the generated molecules for property adherence, and the model updates its parameters to maximize this reward.
- Application Example:
    - Designing molecules with high binding affinity and low synthetic complexity.
 
  
### 5.4.5 Representations and Conditioning Strategies

Effective molecular property optimization using conditional generative models depends heavily on the choice of molecular representation and the encoding of property conditions. This section explores the key strategies for representing molecules and encoding conditions, along with the trade-offs between representation complexity and model performance.

#### 5.4.5.1 Representing Molecules for Generative Models

Different molecular representations influence the performance and efficiency of generative models. The most commonly used representations are:



- SMILES Strings:
    - A linear textual representation of molecular structures.
    - Simple to preprocess and widely used for sequence-based models like RNNs, VAEs, and transformers.
    - Advantages:
        - Easy to work with using standard NLP techniques.
        - Compact and human-readable.
    - Challenges:
        - Sensitive to small changes (e.g., one character change can drastically alter the molecule).
        - Does not explicitly capture 3D structural information.
- Molecular Graphs:
    - Represent molecules as graphs, with atoms as nodes and bonds as edges.
    - Suitable for graph neural networks (GNNs) or graph-based generative models.
    - Advantages:
        - Captures the underlying structure and relationships between atoms.
        - More robust for tasks involving structural optimization.
    - Challenges:
        - Higher computational complexity compared to SMILES.
        - Requires specialized graph-based architectures.
- 3D Structures:
    - Include spatial coordinates of atoms, representing molecular conformations in 3D space.
    - Used in models that integrate quantum-chemical calculations or focus on protein-ligand interactions.
    - Advantages:
        - Provides detailed information about molecular geometry and interactions.
        - Crucial for applications in drug binding and docking studies.
    - Challenges:
        - Computationally intensive to handle and process.
    - Requires additional data (e.g., X-ray crystallography or simulations).

#### 5.4.5.2 Encoding Conditions

The conditions used to guide generative models can be scalar (numerical) or categorical, depending on the property to be optimized.
- Scalar Properties:
    - Examples: Solubility, binding affinity, molecular weight.
    - Encoding:
        - Directly use numerical values or normalize them within a specific range.
        - Feed as continuous input to the model or integrate into the latent space.
    - Application: Generate molecules with a specific solubility threshold or within a molecular weight range.
- Categorical Data:
    - Examples: Functional groups, molecule types, or chemical classes.
    - Encoding:
        - Use one-hot encoding, where each category is represented as a binary vector.
        - Alternatively, embed categorical data as dense vectors using an embedding layer.
    - Application: Generate molecules that belong to a specific class (e.g., benzene derivatives) or contain certain functional groups (e.g., hydroxyl groups).

#### 5.4.5.3 Trade-Offs Between Representation Complexity and Model Performance
- Simpler Representations (e.g., SMILES):
    - Easier to preprocess and faster to train.
    - Risk of losing critical structural or spatial information.
- Complex Representations (e.g., molecular graphs, 3D structures):
    - Provide richer information about the molecule.
    - Require more computational resources and specialized architectures.
The choice of representation and encoding depends on the task's requirements, the computational budget, and the complexity of the target properties.

### 5.4.6 Applications in Molecular Property Optimization

Conditional generative models have demonstrated transformative potential across various domains in molecular property optimization. This section highlights key applications, emphasizing their role in accelerating discovery and improving the efficiency of molecular design.

#### 5.4.6.1 Drug Discovery

- Generating Molecules with Specific ADMET Properties:
    - ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties are crucial in drug development. Conditional generative models can produce molecules that satisfy these requirements, ensuring better pharmacokinetics and reduced side effects.
    - Example: A cVAE can be trained to generate molecules with high bioavailability and low toxicity.
- Ligand-Based and Structure-Based Drug Design:
    - Ligand-based models generate molecules based on known active compounds, optimizing specific properties like binding affinity.
    - Structure-based models use 3D information about the target protein to generate molecules that fit its binding pocket.
    - Example: A conditional GAN could generate ligands with optimal binding to a specific enzyme, improving the lead optimization process.

#### 5.4.6.2 Materials Science

- Designing Polymers with Target Mechanical Properties:
    - Polymers with specific mechanical properties, such as high elasticity, strength, or thermal stability, are crucial in materials design.
    - Conditional models allow the generation of polymer candidates that meet predefined criteria, reducing the time required for experimental testing.
    - Example: A transformer-based model can generate polymer structures optimized for tensile strength while maintaining low production costs.
- Thermal Properties Optimization:
    - Generate materials with specific thermal conductivity or resistance to heat, which are essential in electronics or aerospace industries.
    - Example: Use a graph-based conditional model to design materials for heat sinks with high thermal conductivity.

#### 5.4.6.3 Synthetic Chemistry

- Optimizing Molecules for Synthesis Feasibility:
    - Molecules designed through generative models can sometimes be difficult or expensive to synthesize. Conditional models can optimize for synthetic accessibility, ensuring that generated molecules can be produced using available chemical pathways.
    - Example: A cGAN could generate molecules that minimize the number of synthetic steps while maintaining desired properties like high yield.
- Designing Molecules for Specific Reaction Pathways:
    - Conditional models can generate molecules that align with specific reaction mechanisms or catalytic conditions, aiding in reaction design and optimization.
    - Example: Generate precursors for a specific polymerization reaction to produce biodegradable plastics.

### 5.4.7 Generative Models with Conditions Full Manual Process

google colab: [https://colab.research.google.com/drive/1uXjcu_bzygtia9xgEHN76xMmQtcV0sY-?usp=sharing](https://colab.research.google.com/drive/18EQfIEUt72nzruy4_2rb0aAXwglOT5C_?usp=sharing)

**Step 1: Install Required Libraries**
<pre>
    <code class="python">
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
    </code>
</pre>

**Step 2: Download and Upload the ESOL Dataset from Deepchemdata**

The **ESOL Dataset** is available on Github: https://github.com/deepchem/deepchem/blob/master/datasets/delaney-processed.csv

<pre>
    <code class="python">
# Load the ESOL dataset (direct source: https://github.com/deepchem/deepchem/blob/master/datasets/delaney-processed.csv)
url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
data = pd.read_csv(url)

# Display the first few rows of the dataset
data.head()
    </code>
</pre>

**Step 3: Data Preprocessing and Encoding**


<pre>
    <code class="python">
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
    </code>
</pre>


**Step 4: Conditional Variational Autoencoder (cVAE) Architecture**


<pre>
   <code class = "python">
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
   </code>
</pre>

**Step 5: Custom cVAE Model Implementation and Training**
<pre>
   <code class = "python">
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
   </code>
</pre>

**Step 6: Molecule Generation, Evaluation, and Visualization**
<pre>
   <code class = "python">
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
   </code>
</pre>

**Step 7: Latent Space Visualization Using PCA**
<pre>
   <code class = "python">
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
   </code>
</pre>

**Step 8: Latent Space Exploration and Molecule Generation Based on Solubility**
<pre>
   <code class = "python">
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
   </code>
</pre>

**Step 9: Latent Space Targeting for Property-Driven Molecule Generation and Visualization**
<pre>
   <code class = "python">
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
   </code>
</pre>

