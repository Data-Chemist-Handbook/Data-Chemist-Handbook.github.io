---
title: 3. Machine Learning Models
author: Haomin
date: 2024-08-13
category: Jekyll
layout: post
---

Machine learning (ML) is a subfield of artificial intelligence (AI) focused on developing algorithms and statistical models that enable computers to learn from and make decisions based on data. Unlike traditional programming, where explicit instructions are given, machine learning systems identify patterns and insights from large datasets, improving their performance over time through experience.

ML encompasses various techniques, including supervised learning, where models are trained on labeled data to predict outcomes; unsupervised learning, which involves discovering hidden patterns or groupings within unlabeled data; and reinforcement learning, where models learn optimal actions through trial and error in dynamic environments. These methods are applied across diverse domains, from natural language processing and computer vision to recommendation systems and autonomous vehicles, revolutionizing how technology interacts with the world.

## 3.1 Decision Tree and Random Forest

### 3.1.1 Decision Trees

**Decision Trees** are intuitive and powerful models used in machine learning to make predictions and decisions. Think of it like playing a game of 20 questions, where each question helps you narrow down the possibilities. Decision trees function similarly; they break down a complex decision into a series of simpler questions based on the data.

Each question, referred to as a "decision," relies on a specific characteristic or feature of the data. For instance, if you're trying to determine whether a fruit is an apple or an orange, the initial question might be, "Is the fruit's color red or orange?" Depending on the answer, you might follow up with another question---such as, "Is the fruit's size small or large?" This questioning process continues until you narrow it down to a final answer (e.g., the fruit is either an apple or an orange).

In a decision tree, these questions are represented as nodes, and the possible answers lead to different branches. The final outcomes are represented at the end of each branch, known as leaf nodes. One of the key advantages of decision trees is their clarity and ease of understanding---much like a flowchart. However, they can also be prone to overfitting, especially when dealing with complex datasets that have many features. Overfitting occurs when a model performs exceptionally well on training data but fails to generalize to new or unseen data.

In summary, decision trees offer an intuitive approach to making predictions and decisions, but caution is required to prevent them from becoming overly complicated and tailored too closely to the training data.

### 3.1.2 Random Forest

**Random Forests** address the limitations of decision trees by utilizing an ensemble of multiple trees instead of relying on a single one. Imagine you're gathering opinions about a game outcome from a group of people; rather than trusting just one person's guess, you ask everyone and then take the most common answer. This is the essence of how a Random Forest operates.

In a Random Forest, numerous decision trees are constructed, each making its own predictions. However, a key difference is that each tree is built using a different subset of the data and considers different features of the data. This technique, known as bagging (Bootstrap Aggregating), allows each tree to provide a unique perspective, which collectively leads to a more reliable prediction.

When making a final prediction, the Random Forest aggregates the predictions from all the trees. For classification tasks, it employs majority voting to determine the final class label, while for regression tasks, it averages the results.

Random Forests typically outperform individual decision trees because they are less likely to overfit the data. By combining multiple trees, they achieve a balance between model complexity and predictive performance on unseen data.

#### Real-Life Analogy

Consider Andrew, who wants to decide on a destination for his year-long vacation. He starts by asking his close friends for suggestions. The first friend asks Andrew about his past travel preferences, using his answers to recommend a destination. This is akin to a decision tree approach---one friend following a rule-based decision process.

Next, Andrew consults more friends, each of whom poses different questions to gather recommendations. Finally, Andrew chooses the places suggested most frequently by his friends, mirroring the Random Forest algorithm's method of aggregating multiple decision trees' outputs.

### 3.1.3 Implementing Random Forest on the BBBP Dataset

This guide demonstrates how to implement a **Random Forest** algorithm in Python using the **BBBP (Blood–Brain Barrier Permeability)** dataset. The **BBBP dataset** is used in cheminformatics to predict whether a compound can cross the blood-brain barrier based on its chemical structure.

The dataset contains **SMILES** (Simplified Molecular Input Line Entry System) strings representing chemical compounds, and a **target column** that indicates whether the compound is permeable to the blood-brain barrier or not.

The goal is to predict whether a given chemical compound will cross the blood-brain barrier, based on its molecular structure. This guide walks you through downloading the dataset, processing it, and training a **Random Forest** model.

#### Step 1: Install RDKit (Required for SMILES to Fingerprint Conversion)

We need to use the RDKit library, which is essential for converting **SMILES strings** into molecular fingerprints, a numerical representation of the molecule.

```python
# Install the RDKit package via conda-forge
!pip install -q condacolab
import condacolab
condacolab.install()

# Now install RDKit
!mamba install -c conda-forge rdkit -y

# Import RDKit and check if it's installed successfully
from rdkit import Chem
print("RDKit is successfully installed!")
```

#### Step 2: Download the BBBP Dataset from Kaggle

The **BBBP dataset** is hosted on Kaggle, a popular platform for datasets and machine learning competitions. To access the dataset, you need a Kaggle account and an API key for authentication. Here's how you can set it up:

##### Step 2.1: Create a Kaggle Account
1. Visit Kaggle and create an account if you don't already have one.
2. Once you're logged in, go to your profile by clicking on your profile picture in the top right corner, and select My Account.

##### Step 2.2: Set Up the Kaggle API Key
1. Scroll down to the section labeled API on your account page.
2. Click on the button "Create New API Token". This will download a file named kaggle.json to your computer.
3. Keep this file safe! It contains your API key, which you'll use to authenticate when downloading datasets.

##### Step 2.3: Upload the Kaggle API Key
Once you have the kaggle.json file, you need to upload it to your Python environment:

1. If you're using a notebook environment like Google Colab, use the code below to upload the file:

```python
# Upload the kaggle.json file from google.colab import 
files uploaded = files.upload() 
# Move the file to the right directory for authentication 
!mkdir -p ~/.kaggle !mv kaggle.json ~/.kaggle/ !chmod 600 ~/.kaggle/kaggle.json
```

2. If you're using a local Jupyter Notebook:
   Place the kaggle.json file in a folder named .kaggle within your home directory:
   - On Windows: Place it in C:\Users\<YourUsername>\.kaggle.
   - On Mac/Linux: Place it in ~/.kaggle.

##### Step 2.4: Install the Required Libraries
To interact with Kaggle and download the dataset, you need the Kaggle API client. Install it with the following command:

```python
!pip install kaggle
```

##### Step 2.5: Download the BBBP Dataset
Now that the API key is set up, you can download the dataset using the Kaggle API:

```python
# Download the BBBP dataset using the Kaggle API 
!kaggle datasets download -d priyanagda/bbbp-smiles 
# Unzip the downloaded file 
!unzip bbbp-smiles.zip -d bbbp_dataset
```

This code will:
1. Download the dataset into your environment.
2. Extract the dataset files into a folder named bbbp_dataset.

##### Step 2.6: Verify the Download
After downloading, check the dataset files to confirm that everything is in place:

```python
# List the files in the dataset folder 
import os 
dataset_path = "bbbp_dataset" 
files = os.listdir(dataset_path) 
print("Files in the dataset:", files)
```

By following these steps, you will have successfully downloaded and extracted the BBBP dataset, ready for further analysis and processing.

#### Step 3: Load the BBBP Dataset

After downloading the dataset, we'll load the **BBBP dataset** into a **pandas DataFrame**. The dataset contains the **SMILES strings** and the **target variable** (`p_np`), which indicates whether the compound can cross the blood-brain barrier (binary classification: `1` for permeable, `0` for non-permeable).

```python
import pandas as pd

# Load the BBBP dataset (adjust the filename if it's different)
data = pd.read_csv("bbbp.csv")  # Assuming the dataset is named bbbp.csv
print("Dataset Head:", data.head())
```

#### Step 4: Convert SMILES to Molecular Fingerprints

To use the **SMILES strings** for modeling, we need to convert them into **molecular fingerprints**. This process turns the chemical structures into a numerical format that can be fed into machine learning models. We'll use **RDKit** to generate these fingerprints using the **Morgan Fingerprint** method.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Function to convert SMILES to molecular fingerprints
def featurize_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    else:
        return None

# Apply featurization to the dataset
features = [featurize_molecule(smi) for smi in data['smiles']]  # Replace 'smiles' with the actual column name if different
features = [list(fp) if fp is not None else np.zeros(1024) for fp in features]  # Handle missing data by filling with zeros
X = np.array(features)
y = data['p_np']  # Target column (1 for permeable, 0 for non-permeable)
```

The diagram below provides a visual representation of what this code does:

![Smiles Diagram](../../resource/img/random_forest_decision_tree/smiles.png)

*Figure: SMILES to Molecular Fingerprints Conversion Process*

#### Step 5: Split Data into Training and Testing Sets

To evaluate the model, we need to split the data into training and testing sets. The **train_test_split** function from **scikit-learn** will handle this. We'll use 80% of the data for training and 20% for testing.

```python
from sklearn.model_selection import train_test_split

# Split data into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

The diagram below provides a visual representation of what this code does:

![Train Test Split Diagram](../../resource/img/random_forest_decision_tree/train_test_split.png)

*Figure: Data Splitting Process for Training and Testing*

#### Step 6: Train the Random Forest Model

We'll use the **RandomForestClassifier** from **scikit-learn** to build the model. A Random Forest is an ensemble method that uses multiple decision trees to make predictions. The more trees (`n_estimators`) we use, the more robust the model will be, but the longer the model will take to run. For the most part, n_estimators is set to 100 in most versions of scikit-learn. However, for more complex datasets, higher values like 500 or 1000 may improve performance.

```python
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

The diagram below provides a visual explanation of what is going on here:

![Random Forest Decision Tree Diagram](../../resource/img/random_forest_decision_tree/random_forest_diagram.png)

*Figure: Random Forest Algorithm Structure*

#### Step 7: Evaluate the Model

After training the model, we'll use the **test data** to evaluate its performance. We will print the accuracy and the classification report to assess the model's precision, recall, and F1 score.

```python
from sklearn.metrics import accuracy_score, classification_report

# Predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate accuracy and performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:", classification_report(y_test, y_pred))
```

#### Model Performance and Parameters

- **Accuracy**: The proportion of correctly predicted instances out of all instances.
- **Classification Report**: Provides additional metrics like precision, recall, and F1 score.
  
In this case, we achieved an **accuracy score of ~87%**.

**Key Hyperparameters:**
- **n_estimators**: The number of trees in the Random Forest. More trees generally lead to better performance but also require more computational resources.
- **test_size**: The proportion of data used for testing. A larger test size gives a more reliable evaluation but reduces the amount of data used for training.
- **random_state**: Ensures reproducibility by initializing the random number generator to a fixed seed.

#### Conclusion

This guide demonstrated how to implement a Random Forest model to predict the **Blood–Brain Barrier Permeability (BBBP)** using the **BBBP dataset**. By converting **SMILES strings** to molecular fingerprints and using a **Random Forest classifier**, we were able to achieve an accuracy score of around **87%**.

Adjusting parameters like the number of trees (`n_estimators`) or the split ratio (`test_size`) can help improve the model's performance. Feel free to experiment with these parameters and explore other machine learning models for this task!

### 3.1.4 Approaching Random Forest Problems

When tackling a classification or regression problem using the Random Forest algorithm, a systematic approach can enhance your chances of success. Here's a step-by-step guide to effectively solve any Random Forest problem:

1. **Understand the Problem Domain**: Begin by thoroughly understanding the problem you are addressing. Identify the nature of the data and the specific goal—whether it's classification (e.g., predicting categories) or regression (e.g., predicting continuous values). Familiarize yourself with the dataset, including the features (independent variables) and the target variable (dependent variable).

2. **Data Collection and Preprocessing**: Gather the relevant dataset and perform necessary preprocessing steps. This may include handling missing values, encoding categorical variables, normalizing or standardizing numerical features, and removing any outliers. Proper data cleaning ensures that the model learns from quality data.

3. **Exploratory Data Analysis (EDA)**: Conduct an exploratory data analysis to understand the underlying patterns, distributions, and relationships within the data. Visualizations, such as scatter plots, histograms, and correlation matrices, can provide insights that inform feature selection and model tuning.

4. **Feature Selection and Engineering**: Identify the most relevant features for the model. This can be achieved through domain knowledge, statistical tests, or feature importance metrics from preliminary models. Consider creating new features through feature engineering to enhance model performance.

5. **Model Training and Parameter Tuning**: Split the dataset into training and testing sets, typically using an 80-20 or 70-30 ratio. Train the Random Forest model using the training data, adjusting parameters such as the number of trees (`n_estimators`), the maximum depth of the trees (`max_depth`), and the minimum number of samples required to split an internal node (`min_samples_split`). Utilize techniques like grid search or random search to find the optimal hyperparameters.

6. **Model Evaluation**: Once trained, evaluate the model's performance on the test set using appropriate metrics. For classification problems, metrics such as accuracy, precision, recall, F1 score, and ROC-AUC are valuable. For regression tasks, consider metrics like mean absolute error (MAE), mean squared error (MSE), and R-squared.

7. **Interpretation and Insights**: Analyze the model's predictions and feature importance to derive actionable insights. Understanding which features contribute most to the model can guide decision-making and further improvements in the model or data collection.

8. **Iterate and Improve**: Based on the evaluation results, revisit the previous steps to refine your model. This may involve further feature engineering, collecting more data, or experimenting with different algorithms alongside Random Forest to compare performance.

9. **Deployment**: Once satisfied with the model's performance, prepare it for deployment. Ensure the model can process incoming data and make predictions in a real-world setting, and consider implementing monitoring tools to track its performance over time.

By following this structured approach, practitioners can effectively leverage the Random Forest algorithm to solve a wide variety of problems, ensuring thorough analysis, accurate predictions, and actionable insights.

### 3.1.5 Strengths and Weaknesses of Random Forest

**Strengths:**

- **Robustness**: Random Forests are less prone to overfitting compared to individual decision trees, making them more reliable for new data.

- **Versatility**: They can handle both classification and regression tasks effectively.

- **Feature Importance**: Random Forests provide insights into the significance of each feature in making predictions.

**Weaknesses:**

- **Complexity**: The model can become complex, making it less interpretable than single decision trees.

- **Resource Intensive**: Training a large number of trees can require significant computational resources and time.

- **Slower Predictions**: While individual trees are quick to predict, aggregating predictions from multiple trees can slow down the prediction process.

---

### Section 3.1 – Quiz Questions

#### 1) Factual Questions

##### Question 1
What is the primary reason a Decision Tree might perform very well on training data but poorly on new, unseen data?

**A.** Underfitting  
**B.** Data leakage  
**C.** Overfitting  
**D.** Regularization  

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Decision Trees can easily overfit the training data by creating very complex trees that capture noise instead of general patterns. This hurts their performance on unseen data.
</details>

---

##### Question 2
In a Decision Tree, what do the internal nodes represent?

**A.** Possible outcomes  
**B.** Splitting based on a feature  
**C.** Aggregation of multiple trees  
**D.** Random subsets of data  

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Internal nodes represent decision points where the dataset is split based on the value of a specific feature (e.g., "Is the fruit color red or orange?").
</details>

---

##### Question 3
Which of the following best explains the Random Forest algorithm?

**A.** A single complex decision tree trained on all the data  
**B.** Many decision trees trained on identical data to improve depth  
**C.** Many decision trees trained on random subsets of the data and features  
**D.** A clustering algorithm that separates data into groups  

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Random Forests use bagging to train multiple decision trees on different random subsets of the data and different random subsets of features, making the ensemble more robust.
</details>

---

##### Question 4
When training a Random Forest for a **classification task**, how is the final prediction made?

**A.** By taking the median of the outputs  
**B.** By taking the average of probability outputs  
**C.** By majority vote among trees' predictions  
**D.** By selecting the tree with the best accuracy  

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
For classification problems, the Random Forest algorithm uses majority voting — the class most predicted by the individual trees becomes the final prediction.
</details>

---

#### 2) Conceptual Questions

##### Question 5
You are given a dataset containing information about chemical compounds, with many categorical features (such as "molecular class" or "bond type").  
Would using a Random Forest model be appropriate for this dataset?

**A.** No, Random Forests cannot handle categorical data.  
**B.** Yes, Random Forests can naturally handle datasets with categorical variables after encoding.  
**C.** No, Random Forests only work on images.  
**D.** Yes, but only if the dataset has no missing values.

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
Random Forests can handle categorical data after simple preprocessing, such as label encoding or one-hot encoding. They are robust to different feature types, including numerical and categorical.
</details>

---

##### Question 6
Suppose you have your molecule fingerprints stored in variables `X` and your labels (0 or 1 for BBBP) stored in `y`.  
Which of the following correctly splits the data into **80% training** and **20% testing** sets?

**A.**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
```

**B.**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**C.**
```python
X_train, X_test = train_test_split(X, y, test_size=0.8, random_state=42)
```

**D.**
```python
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Explanation:  
In Random Forest modeling, we use train_test_split from sklearn.model_selection.

test_size=0.2 reserves 20% of the data for testing, leaving 80% for training.

The function returns train features, test features, train labels, and test labels — in that exact order:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

A, C, and D are wrong because...

(A) reverses train and test sizing.

(C) mistakenly sets test_size=0.8 (which would leave only 20% for training — wrong).

(D) messes up the return order (train features and labels must come first).
</details>

<details>
<summary>▶ Show Solution Code</summary>
<pre><code class="language-python">
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
</code></pre>
</details>

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

---

## 3.3 Graph Neural Network

**Graph Neural Networks (GNNs)** offer a new and powerful way to handle molecular machine learning. Traditional neural networks are good at working with fixed-size inputs, such as images or sequences. However, molecules are different in nature. They are best represented as **graphs**, where **atoms are nodes** and **chemical bonds are edges**. This kind of graph structure has always been central in chemistry, appearing in everything from simple Lewis structures to complex reaction pathways. GNNs make it possible for computers to work directly with this kind of data structure.

Unlike images or text, molecules do not follow a regular shape or order. This makes it hard for conventional neural networks to process them effectively. Convolutional neural networks (CNNs) are designed for image data, and recurrent neural networks (RNNs) are built for sequences, but neither is suited to the irregular and highly connected structure of molecules. As a result, older models often fail to capture how atoms are truly linked inside a molecule.

Before GNNs were introduced, chemists used what are known as **molecular descriptors**. These are numerical features based on molecular structure, such as how many functional groups a molecule has or how its atoms are arranged in space. These descriptors were used as input for machine learning models. However, they often **lose important information** about the exact way atoms are connected. This loss of detail limits how well the models can predict molecular behavior.

GNNs solve this problem by learning directly from the molecular graph. Instead of relying on handcrafted features, GNNs use the structure itself to learn what matters. Each atom gathers information from its neighbors in the graph, which helps the model understand the molecule as a whole. This approach leads to **more accurate predictions** and also makes the results **easier to interpret**.

In short, GNNs allow researchers to build models that reflect the true structure of molecules. They avoid the limitations of older methods by directly using the connections between atoms, offering a more natural and powerful way to predict molecular properties.

### 3.3.1 What Are Graph Neural Networks?

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/12cDcFUbyWz4ltPVRcWhbXVq8vEZp2o56?usp=sharing)

#### Why Are Molecules Naturally Graphs?

Let's start with the most fundamental question: Why do we say molecules are graphs?

Imagine a water molecule (H₂O). If you've taken chemistry, you know it looks like this:

```
H - O - H
```

This is already a graph! Let's break it down:

- **Nodes (vertices)**: The atoms - one oxygen (O) and two hydrogens (H)
- **Edges (connections)**: The chemical bonds - two O-H bonds

Now consider a slightly more complex molecule - ethanol (drinking alcohol):

```
    H H
    | |
H - C-C - O - H
    | |
    H H
```

Again, we have a graph:

- **9 nodes**: 2 carbons, 1 oxygen, 6 hydrogens
- **8 edges**: All the chemical bonds connecting these atoms

**Here's the key insight**: A molecule's properties depend heavily on how its atoms are connected. Water dissolves salt because its bent O-H-O structure creates a polar molecule. Diamond is hard while graphite is soft - both are pure carbon, but connected differently!

#### The Molecular Property Prediction Challenge

Before GNNs, how did computers predict molecular properties like solubility, toxicity, or drug effectiveness? Scientists would calculate numerical "descriptors" - features like:

- Molecular weight (sum of all atom weights)
- Number of oxygen atoms
- Number of rotatable bonds
- Surface area

But this approach has a fundamental flaw. Consider these two molecules:

```
Molecule A:  H-O-C-C-C-C-O-H     (linear structure)
Molecule B:  H-O-C-C-O-H         (branched structure)
                 |
                 C-C
```

Traditional descriptors might count:

- Both have 2 oxygens ✓
- Both have similar molecular weights ✓
- Both have OH groups ✓

Yet their properties could be vastly different! The traditional approach loses the connectivity information - it treats molecules as "bags of atoms" rather than structured entities.

#### Enter Graph Neural Networks

GNNs solve this problem elegantly. They process molecules as they truly are - graphs where:

| Graph Component | Chemistry Equivalent | Example in Ethanol               |
| --------------- | -------------------- | -------------------------------- |
| Node            | Atom                 | C, C, O, H, H, H, H, H, H        |
| Edge            | Chemical bond        | C-C, C-O, C-H bonds              |
| Node features   | Atomic properties    | Carbon has 4 bonds, Oxygen has 2 |
| Edge features   | Bond properties      | Single bond, double bond         |
| Graph           | Complete molecule    | The entire ethanol structure     |

#### How GNNs Learn from Molecular Graphs

The magic of GNNs lies in **message passing** - atoms "talk" to their neighbors through bonds. Let's see how this works step by step:

**Step 1: Initial State**
Each atom starts knowing only about itself:

```
Carbon-1: "I'm carbon with 4 bonds"
Carbon-2: "I'm carbon with 4 bonds"  
Oxygen:   "I'm oxygen with 2 bonds"
```

**Step 2: First Message Pass**
Atoms share information with neighbors:

```
Carbon-1: "I'm carbon connected to another carbon and 3 hydrogens"
Carbon-2: "I'm carbon between another carbon and an oxygen"
Oxygen:   "I'm oxygen connected to a carbon and a hydrogen"
```

**Step 3: Second Message Pass**
Information spreads further:

```
Carbon-1: "I'm in an ethyl group (CH3CH2-)"
Carbon-2: "I'm the connection point to an OH group"
Oxygen:   "I'm part of an alcohol (-OH) group"
```

After enough message passing, each atom understands its role in the entire molecular structure!

#### Why Molecular Property Prediction Matters

Molecular property prediction is at the heart of modern drug discovery and materials science. Consider these real-world applications:

1. **Drug Discovery**: Will this molecule pass through the blood-brain barrier?
2. **Environmental Science**: How long will this chemical persist in water?
3. **Materials Design**: What's the melting point of this new polymer?

Traditional experiments to measure these properties are expensive and time-consuming. If we can predict properties from structure alone, we can:

- Screen millions of virtual compounds before synthesizing any
- Identify promising drug candidates faster
- Avoid creating harmful compounds

Let's implement a simple example to see how we represent molecules as graphs in code.

We’ll walk step-by-step through a basic molecular graph construction pipeline using **RDKit**, a popular cheminformatics toolkit in Python. You’ll learn how to load molecules, add hydrogens, inspect atoms and bonds, and prepare graph-based inputs for further learning.

---

#### 1. Load a molecule and include hydrogen atoms

To start, we need to load a molecule using **RDKit**. RDKit provides a function `Chem.MolFromSmiles()` to create a molecule object from a **SMILES string** (a standard text representation of molecules). However, by default, hydrogen atoms are not included explicitly in the molecule. To use GNNs effectively, we want **all atoms explicitly shown**, so we also call `Chem.AddHs()` to add them in.

Let’s break down the functions we’ll use:

* `Chem.MolFromSmiles(smiles_str)`:
  Creates an `rdkit.Chem.rdchem.Mol` object from a SMILES string. This object represents the molecule internally as atoms and bonds.

* `mol.GetNumAtoms()`:
  Returns the number of atoms *currently present* in the molecule object (by default, RDKit does not include H atoms unless you explicitly add them).

* `Chem.AddHs(mol)`:
  Returns a new molecule object with **explicit hydrogen atoms** added to the input `mol`.

<details>
<summary>▶ Click to see code: Basic molecule to graph conversion</summary>
<pre><code class="language-python">
from rdkit import Chem
import numpy as np

# Step 1: Create a molecule object from the SMILES string for water ("O" means one oxygen atom)
water = Chem.MolFromSmiles("O")

# Step 2: Count how many atoms are present (will be 1 — only the oxygen)
print(f"Number of atoms: {water.GetNumAtoms()}")  # Output: 1

# Step 3: Add explicit hydrogen atoms
water = Chem.AddHs(water)

# Step 4: Count again — now we should see 3 atoms (1 O + 2 H)
print(f"Number of atoms with H: {water.GetNumAtoms()}")  # Output: 3
</code></pre>
</details>

---

#### 2. Access the bond structure (graph edges)

Once we have the molecule, we want to know **which atoms are connected**—this is the basis for constructing a graph. RDKit stores this as a list of `Bond` objects, which we can retrieve using `mol.GetBonds()`.

Let’s break down the functions used here:

* `mol.GetBonds()`:
  Returns a list of **bond objects** in the molecule. Each bond connects two atoms.

* `bond.GetBeginAtomIdx()` and `bond.GetEndAtomIdx()`:
  These return the **indices** (integers) of the two atoms that are connected by the bond.

* `mol.GetAtomWithIdx(idx).GetSymbol()`:
  This retrieves the **chemical symbol** (e.g. "H", "O") of the atom at a given index.

<details>
<summary>▶ Click to see code: Extracting graph connectivity</summary>
<pre><code class="language-python">
# Print all bonds in the molecule in the form: Atom(index) -- Atom(index)
print("Water molecule connections:")
for bond in water.GetBonds():
    atom1_idx = bond.GetBeginAtomIdx()  # e.g., 0
    atom2_idx = bond.GetEndAtomIdx()    # e.g., 1
    atom1 = water.GetAtomWithIdx(atom1_idx).GetSymbol()  # e.g., "O"
    atom2 = water.GetAtomWithIdx(atom2_idx).GetSymbol()  # e.g., "H"
    print(f"  {atom1}({atom1_idx}) -- {atom2}({atom2_idx})")

# Output:
# Water molecule connections:
#   O(0) -- H(1)
#   O(0) -- H(2)
</code></pre>
</details>

---

#### 3. Extract simple atom-level features

Each atom will become a **node** in our graph, and we often associate it with a **feature vector**. To keep things simple, we start with just the **atomic number**.

Here’s what each function does:

* `atom.GetAtomicNum()`:
  Returns the **atomic number** (integer) for the element, e.g., 1 for hydrogen, 8 for oxygen.

* `mol.GetAtoms()`:
  Returns a generator over all `Atom` objects in the molecule.

* `atom.GetSymbol()`:
  Returns the chemical symbol ("H", "O", etc.), useful for printing/debugging.

<details>
<summary>▶ Click to see code: Atom feature extraction</summary>
<pre><code class="language-python">
# For each atom, we print its atomic number
def get_atom_features(atom):
    # Atomic number is a simple feature used in many models
    return [atom.GetAtomicNum()]

# Apply to all atoms in the molecule
for i, atom in enumerate(water.GetAtoms()):
    features = get_atom_features(atom)
    symbol = atom.GetSymbol()
    print(f"Atom {i} ({symbol}): features = {features}")

# Output
# Atom 0 (O): features = [8]
# Atom 1 (H): features = [1]
# Atom 2 (H): features = [1]
</code></pre>
</details>


---

#### 4. Build the undirected edge list

Now we extract the **list of bonds as pairs of indices**. Since GNNs typically use **undirected graphs**, we store each bond in both directions (i → j and j → i).

Functions involved:

* `bond.GetBeginAtomIdx()`, `bond.GetEndAtomIdx()` (as above)
* We simply collect `[i, j]` and `[j, i]` into a list of edges.

<details>
<summary>▶ Click to see code: Edge extraction</summary>
<pre><code class="language-python">
def get_edge_list(mol):
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])  # undirected graph: both directions
    return edges

# Run on water molecule
water_edges = get_edge_list(water)
print("Water edges:", water_edges)

# Output
# Water edges: [[0, 1], [1, 0], [0, 2], [2, 0]]
</code></pre>
</details>

Each pair represents one connection (bond) between atoms. Including both directions ensures that during **message passing**, information can flow freely from each node to all its neighbors.

---

#### Summary: The Power of Molecular Graphs

Let's recap what we've learned:

1. **Molecules are naturally graphs** - atoms are nodes, bonds are edges
2. **Traditional methods lose structural information** - they treat molecules as bags of features
3. **GNNs preserve molecular structure** - they process the actual connectivity
4. **Message passing allows context learning** - atoms learn from their chemical environment
5. **Property prediction becomes structure learning** - the model learns which structural patterns lead to which properties

In the next section, we'll dive deep into how message passing actually works, building our understanding step by step until we can implement a full molecular property predictor.

### 3.3.2 Message Passing and Graph Convolutions

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1SEcW4uvcI4aTipkP5t9IXS6VxQfQ2tA0?usp=sharing)

At the core of a Graph Neural Network (GNN) is the idea of **message passing**.
The goal is to simulate an important phenomenon in chemistry: **how electronic effects propagate through molecular structures via chemical bonds**. This is something that happens in real molecules, and GNNs try to mimic it through mathematical and computational means.

Let’s first look at a chemistry example.
When a **fluorine atom** is added to a molecule, its **high electronegativity** doesn’t just affect the atom it is directly bonded to. It causes that **carbon atom** to become slightly positive, which in turn affects its other bonds, and so on. The effect ripples outward through the structure.

This is exactly the kind of **structural propagation** that message passing in GNNs is designed to model.

#### The structure of message passing: what happens at each GNN layer?

Even though the idea sounds intuitive, we need a well-defined set of mathematical steps for the computer to execute.
In a GNN, each layer usually follows **three standard steps**:

#### Step 1: **Message Construction**

For every node $i$, we consider all its neighbors $j$ and create a message $m_{ij}$ to describe what information node $j$ wants to send to node $i$.

This message often includes:

* Information about **node $j$** itself
* Information about the **bond between $i$ and $j$** (e.g., single, double, aromatic)

Importantly, we don’t just pass raw features. Instead, we use **learnable functions** (like neural networks) to transform the input into something more meaningful for the task.

#### Step 2: **Message Aggregation**

Once node $i$ receives messages from all neighbors, it aggregates them into a single combined message $m_i$.

The simplest aggregation method is to **sum all incoming messages**:

$$
m_i = \sum_{j \in N(i)} m_{ij}
$$

Here, $N(i)$ is the set of all neighbors of node $i$.
This step is like saying: "I listen to all my neighbors and combine what they told me."

However, in real chemistry, **not all neighbors are equally important**:

* A **double bond** may influence differently than a single bond
* An **oxygen atom** might carry more weight than a hydrogen atom

That’s why advanced GNNs often use **weighted aggregation** or **attention mechanisms** to adjust how each neighbor contributes.

#### Step 3: **State Update**

Finally, node $i$ uses two inputs:

* Its current feature vector $h_i^{(t)}$
* The aggregated message $m_i$

These are combined to produce an **updated node representation** for the next layer:

$$
h_i^{(t+1)} = \text{Update}(h_i^{(t)}, m_i)
$$

This update is usually implemented with a small neural network, such as a **multilayer perceptron (MLP)**. It learns how to combine a node’s old information with the new input from its neighbors to produce something more useful.

In summary, at each GNN layer, **every atom (node) listens to its neighbors and updates its understanding of the molecule**.
After several layers of message passing, each node's embedding captures not just its local features, but also the broader context of the molecular structure.

---

#### Graph Convolutions: Making It Concrete

The term **"graph convolution"** comes from analogy with Convolutional Neural Networks (CNNs) in computer vision. In CNNs, filters slide over local neighborhoods of pixels. In GNNs, we also aggregate information from "neighbors", but now **neighbors are defined by molecular or structural connectivity, not spatial proximity**.

In Graph Convolutional Networks (GCNs), message passing is defined by the following steps at each layer:

$$
h_i^{(t+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{d_i d_j}} W h_j^{(t)} \right)
$$

* $h_i^{(t)}$: Feature of node $i$ at layer $t$
* $W$: Learnable weight matrix
* $d_i$: Degree (number of neighbors) of node $i$
* $\sigma$: Activation function (e.g. ReLU)
* This formula **averages and transforms** neighbor features while normalizing based on node degrees.

In PyTorch Geometric (PyG), the most basic GNN implementation is `GCNConv`. Let’s go through each part of the code.

**PyTorch Geometric Components**

| Component                            | Purpose                                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `torch.tensor(...)`                  | Creates dense tensors (like NumPy arrays) for node features or edge indices.                      |
| `x`                                  | Node feature matrix. Shape = `[num_nodes, num_node_features]`                                     |
| `edge_index`                         | Edge list in **COO format**: `[2, num_edges]`. First row: source nodes. Second row: target nodes. |
| `torch_geometric.data.Data`          | Creates a graph object holding `x`, `edge_index`, and optionally edge/node labels.                |
| `GCNConv(in_channels, out_channels)` | A GCN layer that does: message passing + aggregation + update.                                    |
| `conv(x, edge_index)`                | Applies one layer of graph convolution and returns updated node features.                         |

<details>
<summary>▶ Click to see code</summary>
<pre><code class="language-python">
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define node features: 4 nodes, each with 3 features (e.g., atom types)
x = torch.tensor([
    [1, 0, 0],  # Node 0
    [0, 1, 0],  # Node 1
    [1, 1, 0],  # Node 2
    [0, 0, 1]   # Node 3
], dtype=torch.float)

# Define edges: undirected graph, so each edge appears twice (i → j and j → i)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 0],  # Source nodes
    [1, 0, 2, 1, 3, 2, 0, 3]   # Target nodes
], dtype=torch.long)

# Build the graph using PyG's Data structure
data = Data(x=x, edge_index=edge_index)

# Define a Graph Convolutional Network (GCN) layer:
# input_dim = 3 (features), output_dim = 2
conv = GCNConv(in_channels=3, out_channels=2)

# Apply the GCN (i.e., message passing)
output = conv(data.x, data.edge_index)

# Print updated node features
print("Updated Node Features After Message Passing:")
print(output)
</code></pre>
</details>

This code outputs a tensor of shape `[4, 2]` — one **updated node representation** per node, after applying the GCN layer. For example:

```
Updated Node Features After Message Passing:
tensor([[ 0.2851, -0.0017],
        [ 0.6568, -0.4519],
        [ 0.6180,  0.1266],
        [ 0.2807, -0.3559]], grad_fn=<AddBackward0>)
```

**Variants of Graph Convolutions**

Different GNN models define the message passing process differently:

**Graph Convolutional Networks (GCNs)**
Use simple averaging with normalization. Very stable and interpretable. Good for small graphs with clean structure.

**GraphSAGE**
Introduces neighbor **sampling**, which makes it scalable to large graphs. You can also choose the aggregation function (mean, max, LSTM, etc.).

**Graph Attention Networks (GATs)**
Use attention to assign **different weights** to different neighbors. This is very helpful in chemistry, where some bonds are more important (e.g. polar bonds).

**Message Passing Neural Networks (MPNNs)**
A general and expressive framework. Can use **edge features**, which is important in molecules (e.g. bond type, aromaticity). Many SOTA chemistry models (e.g., D-MPNN) are built on this.

#### Chemical Intuition Behind Message Passing

To understand how message passing in graph neural networks actually captures chemical effects, let’s walk through a concrete example: the molecule **para-nitrophenol**, which features two chemically distinct groups — a nitro group (NO₂) and a hydroxyl group (OH) — placed at opposite ends of a benzene ring.

Chemically speaking, this setup forms a classic "push-pull" system: the nitro group is strongly electron-withdrawing, while the hydroxyl group is electron-donating. This dynamic tension in electron distribution plays a key role in determining the molecule’s acidity, reactivity, and overall behavior. The power of message passing lies in its ability to gradually capture this electron flow, layer by layer.

**GNN Message Passing: Neighborhood Expansion per Layer**
- Layer 0: Self (Each atom only knows its own features)
  - e.g., O knows it's oxygen; N knows it's nitrogen

- Layer 1: 1-hop neighbors (Directly bonded atoms)
  - O learns about the carbon it's attached to
  - N in NO₂ learns about its adjacent O atoms

- Layer 2: 2-hop neighbors (Neighbors of neighbors)
  - O learns about atoms bonded to its neighboring carbon
  - Benzene carbons begin to capture influence from NO₂ and OH

- Layer 3: 3-hop neighborhood (Extended molecular context)
  - Carbons across the ring begin to "feel" opposing substituent effects
  - Push-pull interactions emerge in representation

- Layer 4+: Global context (Full molecule representation)
  - Every atom integrates information from the entire molecule
  - Final features encode global electronic and structural effects

At the initial step (layer 0), each atom only knows itself: for example, the nitrogen in the nitro group knows it is positively charged, the hydroxyl oxygen knows it’s bonded to a hydrogen, and the aromatic carbons know their local type. No context is shared yet.

In the first message passing layer, atoms begin exchanging information with their direct neighbors. The nitro nitrogen learns it is connected to two electron-withdrawing oxygens. The ortho carbon next to the nitro group receives this message and begins to "realize" it’s adjacent to a strong puller. On the other side of the ring, the carbon bonded to the hydroxyl group starts picking up signals from the electron-donating OH.

By the second layer, this influence propagates further. Carbons that are not directly attached to NO₂ or OH now begin receiving mixed signals. The meta carbons, for instance, integrate messages from both sides — they now reflect the competing effects of withdrawal and donation. The model begins to reconstruct the kind of electronic delocalization and inductive influence that chemists would traditionally describe with resonance structures or Hammett constants.

As we add more layers, each atom eventually integrates information from the entire molecule. At this point, the model can “understand” the global distribution of electronic effects. It learns that this is a conjugated aromatic system with competing substituents and adjusts each atom’s representation accordingly. This is not just structural information, but **functional** understanding — the kind that predicts acid dissociation, reactivity sites, or electronic transition behavior.

From a computational point of view, each additional GNN layer expands an atom’s receptive field by one hop. By the third or fourth layer, the atom embeddings encode not only local geometry but also the broader chemical environment. Modern architectures like GAT can further modulate this process by weighting important neighbors more heavily (e.g., polar bonds, charged atoms), and MPNNs can include edge features like bond types or bond orders to enrich the message content.

In short, message passing enables the model to capture what chemists know intuitively: that an atom’s behavior is shaped not just by its identity, but by its **context** within the molecule. This is how GNNs bridge raw structure and chemical insight — atom by atom, bond by bond.

#### Summary

In this section, we explored **message passing** and **graph convolutions**, the fundamental mechanisms that enable Graph Neural Networks to learn from molecular structures. The key insight is that GNNs mimic how electronic effects propagate through chemical bonds in real molecules.

The message passing framework follows three standard steps at each layer:
1. **Message Construction**: Nodes create messages for their neighbors using learnable functions
2. **Message Aggregation**: Each node combines incoming messages (via sum, attention, etc.)
3. **State Update**: Nodes update their representations by combining current features with aggregated messages

Through the lens of para-nitrophenol, we saw how this iterative process gradually expands each atom's "awareness" from local to global context. Starting with only self-knowledge, atoms progressively integrate information from direct neighbors, then second-hop neighbors, until eventually capturing full molecular context including competing electronic effects.

Different GNN architectures (GCN, GraphSAGE, GAT, MPNN) offer various approaches to this process, each with distinct advantages for chemical applications. The choice depends on factors like molecular size, importance of edge features, and computational constraints.

The power of message passing lies in its ability to bridge **structure and function**. By allowing atoms to "communicate" through bonds, GNNs learn representations that encode not just molecular topology, but also the chemical behaviors that emerge from that structure — acidity, reactivity, electronic distribution, and more. This makes GNNs particularly well-suited for molecular property prediction and drug discovery tasks where understanding chemical context is crucial.

### 3.3.3 GNNs for Molecular Property Prediction

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1IemDJyiQuDwBK-iTkaHBgqqfAiM065_b?usp=sharing)

The true test of any machine learning approach lies in its ability to solve real-world problems. For Graph Neural Networks in chemistry, this means predicting molecular properties - from simple physical characteristics like boiling point to complex biological activities like drug efficacy. In this section, we'll build a complete GNN system for molecular property prediction, using water solubility as our target property.

#### The Challenge of Molecular Property Prediction

Predicting how molecules behave in the real world is one of chemistry's grand challenges. Consider water solubility - a property that seems simple but emerges from a complex interplay of factors:

- **Molecular size**: Larger molecules generally dissolve less readily
- **Polarity**: Polar groups like -OH and -NH₂ increase water solubility  
- **Hydrogen bonding**: Both donors and acceptors affect solubility
- **Molecular shape**: Compact molecules pack better in water than extended ones
- **Aromatic systems**: These tend to decrease solubility due to their hydrophobic nature

Traditional approaches to solubility prediction relied on empirical rules like "like dissolves like" or complex equations with dozens of parameters. But these methods often fail for novel molecular structures or struggle to capture subtle effects. GNNs offer a fundamentally different approach: learn the structure-property relationship directly from data.

#### Why Graphs for Molecules?

To appreciate why GNNs excel at molecular property prediction, let's consider how molecules are traditionally represented in machine learning:

**Molecular Fingerprints**: These encode molecular structure as fixed-length bit vectors. While useful, they lose precise connectivity information. Two very different molecules might have similar fingerprints, or subtle but important differences might be obscured.

**Molecular Descriptors**: Calculated properties like molecular weight, logP, or polar surface area. These are interpretable but limited - they're human-designed features that might miss important patterns.

**SMILES Strings**: Text representations that can be processed by sequence models. But SMILES treats molecules as linear sequences, losing the natural graph structure.

In contrast, GNNs work with molecules as they truly are - graphs where atoms are nodes and bonds are edges. This preserves all structural information while allowing the model to learn what matters for the property at hand.

#### Setting Up the Environment

Before we begin building our GNN-based molecular property prediction system, we need to load a set of specialized Python libraries. These libraries fall into several functional categories:

* **Deep Learning Core (PyTorch)**: We use PyTorch to define and train neural networks.
* **Graph Neural Network Layers (PyTorch Geometric)**: These enable message passing and graph pooling, essential for processing molecular graphs.
* **Chemical Representation (RDKit)**: RDKit provides SMILES parsing, hydrogen addition, and graph connectivity extraction from molecules.
* **Data Processing and Visualization**: Standard tools like `pandas`, `numpy`, and `matplotlib` are used for loading data and plotting results.
* **Model Evaluation**: Scikit-learn's metrics help assess our model's performance.
* **I/O Utilities**: We use `requests` and `io` to fetch and process online datasets.

These tools collectively allow us to transform raw molecular strings into structured graph data, feed them into a GNN, and evaluate the model's predictive power.

<details>
<summary>▶ Click to see code: Import necessary libraries</summary>
<pre><code class="language-python">
# PyTorch: Core deep learning library
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric: GNN-specific operations
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# RDKit: Chemistry toolkit for molecular structures  
from rdkit import Chem

# Standard numerical/data libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# I/O utilities
import requests
import io

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
</code></pre>
</details>

This environment gives us everything needed to build, train, and evaluate a molecular graph neural network. Once initialized, we'll move on to data loading and molecular graph construction.

#### Understanding Molecular Data: The ESOL Dataset

For our example, we'll use the ESOL (Estimated SOLubility) dataset – a carefully curated collection of 1,128 molecules with measured aqueous solubility values. This dataset has become a standard benchmark because it's large enough to train meaningful models yet small enough to experiment with quickly.

**Why ESOL for learning GNNs?**
- The dataset covers a wide range of molecular structures
- Solubility values span over 13 log units (a 10 trillion-fold difference!)
- It includes diverse functional groups that affect solubility differently
- The relatively small size allows for quick experimentation

<details>
<summary>▶ Click to see code: Loading the ESOL dataset</summary>
<pre><code class="language-python">
# Load the ESOL dataset from an online CSV file
# We use the requests module to download the file and pandas to parse it.
import requests
import pandas as pd
import io

url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
response = requests.get(url)
data = pd.read_csv(io.StringIO(response.text))

# Extract SMILES strings and target solubility values
smiles_list = data['smiles'].tolist()
solubility_values = data['measured log solubility in mols per litre'].tolist()

print(f"Dataset contains {len(smiles_list)} molecules")
print(f"Solubility range: {min(solubility_values):.2f} to {max(solubility_values):.2f} log S")
</code></pre>
</details>

Let's examine a few example molecules to understand the diversity of the dataset:

<details>
<summary>▶ Click to see code: Exploring example molecules</summary>
<pre><code class="language-python">
# Display a few examples to understand what the data looks like
print("\nExample molecules:")
for i in range(3):
    print(f"  {smiles_list[i]}: {solubility_values[i]:.2f} log S")
</code></pre>
</details>

**Results**
```
Dataset contains 1128 molecules  
Solubility range: -11.60 to 1.58 log S

Example molecules:
    OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O : -0.77 log S
    Cc1occc1C(=O)Nc2ccccc2 : -3.30 log S
    CC(C)=CCCC(C)=CC(=O) : -2.06 log S
```

This dataset covers a wide dynamic range of water solubility. The log S values span more than 13 units, corresponding to a solubility difference of over 10 trillion times between the most and least soluble compounds. This range challenges the model to generalize across diverse structural and chemical properties.

#### Converting Molecules to Graph Representations

To use GNNs, we must first convert molecular structures into graph form. In a molecular graph:
- **Nodes** represent atoms
- **Edges** represent chemical bonds

We'll use RDKit to extract this information.

##### Atom Features

Each atom must be encoded into a numerical vector capturing its basic properties. This vector serves as the node feature in the GNN.

**Chemical motivation for atom features:**
- **Atomic number**: Different elements have vastly different chemical properties
- **Degree**: The number of bonds affects an atom's steric environment and reactivity
- **Formal charge**: Charged atoms strongly influence solubility
- **Aromaticity**: Aromatic atoms participate in delocalized π-electron systems
- **Hydrogen count**: Critical for hydrogen bonding capability

<details>
<summary>▶ Click to see code: Defining atom feature extraction</summary>
<pre><code class="language-python">
from rdkit import Chem

# Define a function to extract chemical features from an atom
# These features are chosen for their relevance to chemical reactivity and structure
def get_atom_features(atom):
    features = [
        atom.GetAtomicNum(),        # Atomic number (C=6, N=7, O=8, etc.)
        atom.GetDegree(),           # Number of directly bonded atoms
        atom.GetFormalCharge(),     # Formal electric charge
        int(atom.GetIsAromatic()),  # Is the atom part of an aromatic ring?
        atom.GetTotalNumHs()        # Number of hydrogen atoms bonded
    ]
    return features
</code></pre>
</details>

Let's test this function on a simple molecule to see what features we extract:

<details>
<summary>▶ Click to see code: Testing atom features on water</summary>
<pre><code class="language-python">
# Test on water (H2O)
water = Chem.MolFromSmiles("O")
water = Chem.AddHs(water)  # Add hydrogen atoms explicitly

print("Water atom features:")
for i, atom in enumerate(water.GetAtoms()):
    features = get_atom_features(atom)
    print(f"  {atom.GetSymbol()}: {features}")
</code></pre>
</details>

**Results**
```
Water atom features:
  O: [8, 2, 0, 0, 0]
  H: [1, 1, 0, 0, 0]
  H: [1, 1, 0, 0, 0]
```

In this example:
* The oxygen atom has atomic number 8, is bonded to 2 hydrogens, and is non-aromatic
* Each hydrogen has atomic number 1 and is bonded to 1 atom (the oxygen)
* These feature vectors form the **node features** for the GNN

##### Bond Connectivity

To complete the molecular graph, we also need the edges – the bonds that connect pairs of atoms.

**Why bidirectional edges?**
In chemistry, bonds represent electron sharing between atoms. The influence flows both ways: an electronegative atom pulls electron density from its neighbors, while electron-rich groups push density outward. By representing each bond twice (i→j and j→i), we allow the GNN to model this bidirectional flow of electronic effects.

<details>
<summary>▶ Click to see code: Extracting bond connectivity</summary>
<pre><code class="language-python">
from rdkit import Chem

# Extract pairwise bond connections as edge list for graph representation
# Each bond is stored twice (i->j and j->i) for undirected graph processing
def get_bond_connections(mol):
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
    return edges
</code></pre>
</details>

Let's visualize how this works on a simple organic molecule:

<details>
<summary>▶ Click to see code: Testing bond extraction on ethanol</summary>
<pre><code class="language-python">
# Test on ethanol (CH3CH2OH)
ethanol = Chem.MolFromSmiles("CCO")
ethanol = Chem.AddHs(ethanol)

connections = get_bond_connections(ethanol)
print(f"Ethanol has {ethanol.GetNumAtoms()} atoms and {len(connections)//2} bonds")
</code></pre>
</details>

**Results**
```
Ethanol has 9 atoms and 8 bonds
```

This confirms that the fully hydrogenated ethanol molecule contains 9 atoms and 8 bonds (16 directed edges for GNN input). Each bond appears **twice** to support **bidirectional message passing**, a key mechanism in GNN architectures.

#### The Graph Neural Network Architecture

Now we'll define the GNN model that will predict molecular properties based on graph structure. The model has three main building blocks:

1. **Graph convolutional layers** – to propagate information across atoms and bonds
2. **Global pooling** – to summarize the whole molecule
3. **Prediction head** – to output a numerical property value

First, let's understand the key components we'll use:

**GCNConv**: A graph convolution layer that updates each atom's features based on its neighbors
**ReLU**: Non-linear activation function that allows the network to learn complex patterns
**global_mean_pool**: Averages all atom features to get a single molecular representation

<details>
<summary>▶ Click to see code: Import GNN components</summary>
<pre><code class="language-python">
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
</code></pre>
</details>

Now let's build our molecular GNN class. We'll start with the initialization:

<details>
<summary>▶ Click to see code: GNN class initialization</summary>
<pre><code class="language-python">
class MolecularGNN(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=3):
        """
        Initialize a GNN for molecular property prediction.
        
        Args:
            num_features (int): Number of input features per atom
            hidden_dim (int): Size of hidden representations  
            num_layers (int): Number of message passing rounds
        """
        super(MolecularGNN, self).__init__()
        
        # Create a list of GCN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer: map input features to hidden dim
        self.gnn_layers.append(GCNConv(num_features, hidden_dim))
        
        # Intermediate layers: message passing
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Final prediction layer: hidden vector → scalar output
        self.predictor = nn.Linear(hidden_dim, 1)
</code></pre>
</details>

**Why this architecture?**
- **Multiple GCN layers**: Allow information to propagate across multiple bonds
- **Hidden dimension of 64**: Balance between expressiveness and overfitting
- **Linear predictor**: Simple mapping from molecular representation to property

Next, we define the forward pass - how data flows through the network:

<details>
<summary>▶ Click to see code: GNN forward pass</summary>
<pre><code class="language-python">
    def forward(self, x, edge_index, batch):
        """
        Forward pass through the GNN.
        
        Args:
            x (Tensor): Node feature matrix
            edge_index (Tensor): Graph connectivity (edges)
            batch (Tensor): Batch vector assigning nodes to graphs
            
        Returns:
            Tensor: Predicted scalar property (e.g., solubility)
        """
        # Apply graph convolutions with ReLU activation
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        # Pool node features to get graph-level embedding
        graph_embedding = global_mean_pool(x, batch)
        
        # Predict property from graph embedding
        return self.predictor(graph_embedding).squeeze(-1)
</code></pre>
</details>

**Core ideas:**
* Each GCN layer performs message passing — updating each atom's features based on neighbors
* ReLU activation introduces non-linearity for learning complex patterns
* Global mean pooling creates a fixed-size representation regardless of molecule size
* The predictor maps this representation to a solubility value

#### Converting Molecules to PyTorch Geometric Graphs

To train a GNN on molecular data, we must convert each molecule from its SMILES string into a graph structure compatible with PyTorch Geometric. This involves:

1. Parsing the SMILES string using RDKit
2. Adding explicit hydrogen atoms (important for accurate chemistry)
3. Extracting atom features and bond connectivity
4. Creating a PyTorch Geometric `Data` object

<details>
<summary>▶ Click to see code: SMILES to graph conversion</summary>
<pre><code class="language-python">
from rdkit import Chem
import torch
from torch_geometric.data import Data

def molecule_to_graph(smiles, solubility=None):
    """
    Convert a SMILES string to a PyTorch Geometric graph.
    
    Args:
        smiles: String representation of the molecule
        solubility: Optional float value of log S for supervised learning
        
    Returns:
        PyTorch Geometric Data object with x (node features),
        edge_index (bond connections), and optional y (label)
    """
    # Parse SMILES and check validity
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogen atoms explicitly
    mol = Chem.AddHs(mol)
    
    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)
</code></pre>
</details>

Now we handle the edge connectivity:

<details>
<summary>▶ Click to see code: Creating edge index and Data object</summary>
<pre><code class="language-python">
    # Extract bond connections
    edge_list = get_bond_connections(mol)
    
    # Handle molecules with no bonds (single atoms)
    if len(edge_list) == 0:
        edge_list = [[0, 0]]  # Self-loop for isolated atoms
    
    # Convert to PyTorch format
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index)
    
    # Add target value if provided
    if solubility is not None:
        data.y = torch.tensor([solubility], dtype=torch.float)
    
    return data
</code></pre>
</details>

Let's test this conversion:

<details>
<summary>▶ Click to see code: Testing molecule to graph conversion</summary>
<pre><code class="language-python">
# Quick test
test_graph = molecule_to_graph("CCO", -0.77)
print(f"Ethanol graph: {test_graph.x.shape[0]} atoms, "
      f"{test_graph.edge_index.shape[1] // 2} bonds")
</code></pre>
</details>

**Explanation:**
* `Chem.AddHs` makes hydrogen atoms explicit for consistency
* The edge list is transposed (`.t()`) to match PyTorch Geometric's format
* `.contiguous()` ensures memory layout is optimized for GPU operations

#### Preparing the Dataset

Now we'll convert all molecules in our dataset to graphs and prepare them for training. We'll use the first 1000 molecules for faster experimentation:

<details>
<summary>▶ Click to see code: Converting dataset to graphs</summary>
<pre><code class="language-python">
from torch_geometric.loader import DataLoader

# Step 1: Convert molecules to graphs (first 1000 for speed)
graphs = []
for smiles, sol in zip(smiles_list[:1000], solubility_values[:1000]):
    graph = molecule_to_graph(smiles, sol)
    if graph is not None:
        graphs.append(graph)

print(f"Successfully converted {len(graphs)} molecules to graphs")

# Step 2: Train/test split
train_size = int(0.8 * len(graphs))
train_graphs = graphs[:train_size]
test_graphs = graphs[train_size:]

print(f"Training set: {len(train_graphs)} molecules")
print(f"Test set: {len(test_graphs)} molecules")
</code></pre>
</details>

**DataLoader setup:**
The DataLoader automatically batches multiple molecular graphs together. PyTorch Geometric handles variable-sized graphs by:
- Concatenating all node features into one large matrix
- Keeping track of which nodes belong to which molecule using a batch index
- Adjusting edge indices to work with the concatenated representation

<details>
<summary>▶ Click to see code: Creating DataLoaders</summary>
<pre><code class="language-python">
# Step 3: Wrap in data loaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# Step 4: Model setup
model = MolecularGNN(num_features=5, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
</code></pre>
</details>

#### Training the Model

The training process involves:
1. Forward pass: compute predictions
2. Loss calculation: measure prediction error
3. Backward pass: compute gradients
4. Optimizer step: update model parameters

<details>
<summary>▶ Click to see code: Training epoch function</summary>
<pre><code class="language-python">
# Step 5: Define one training epoch
def train_epoch(model, loader, optimizer, criterion):
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch in loader:
        # Zero gradients from previous step
        optimizer.zero_grad()
        
        # Forward pass
        prediction = model(batch.x, batch.edge_index, batch.batch)
        
        # Calculate loss
        loss = criterion(prediction.squeeze(), batch.y)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Track total loss
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)
</code></pre>
</details>

**Key details:**
- `model.train()` enables dropout and batch normalization training behavior
- `batch.batch` tells the model which nodes belong to which molecule
- `.squeeze()` removes extra dimensions to match target shape
- We normalize loss by dataset size for consistent reporting

Now let's train for multiple epochs:

<details>
<summary>▶ Click to see code: Training loop</summary>
<pre><code class="language-python">
# Step 6: Training loop
print("Training GNN on ESOL dataset...")
for epoch in range(50):
    loss = train_epoch(model, train_loader, optimizer, criterion)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.3f}")
</code></pre>
</details>

**Results**
```
Training GNN on ESOL dataset...
Epoch 0: Loss = 7.583
Epoch 10: Loss = 3.824
Epoch 20: Loss = 3.767
Epoch 30: Loss = 3.584
Epoch 40: Loss = 3.396
```

Over 50 epochs, the training loss steadily decreases, showing the model is learning to fit the solubility data. The relatively high starting loss reflects the complexity of the task and the broad range of solubility values.

#### Making Predictions

Once trained, we can use the model to predict solubility for new molecules. The prediction process involves:

1. Converting the SMILES to a graph
2. Creating a batch index (required even for single molecules)
3. Running the forward pass
4. Extracting the scalar prediction

<details>
<summary>▶ Click to see code: Prediction function</summary>
<pre><code class="language-python">
def predict_solubility(smiles, model):
    """Predict solubility for a new molecule."""
    # Convert SMILES to graph
    graph = molecule_to_graph(smiles)
    if graph is None:
        return None
    
    # Create dummy batch index (for a single molecule)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    
    # Run model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model(graph.x, graph.edge_index, batch)
    
    return prediction.item()
</code></pre>
</details>

Let's test on some well-known molecules with different chemical properties:

<details>
<summary>▶ Click to see code: Testing predictions on common molecules</summary>
<pre><code class="language-python">
# Try out the model on some well-known molecules
test_molecules = [
    ("O", "Water"),
    ("CCO", "Ethanol"), 
    ("CC(C)=O", "Acetone"),
    ("c1ccccc1", "Benzene"),
    ("CC(=O)O", "Acetic acid")
]

print("\nPredictions for common molecules:")
for smiles, name in test_molecules:
    pred = predict_solubility(smiles, model)
    print(f"  {name}: {pred:.2f} log S")
</code></pre>
</details>

**Results**
```
Predictions for common molecules:
  Water: 0.32 log S
  Ethanol: -1.59 log S
  Acetone: -1.76 log S
  Benzene: -3.44 log S
  Acetic acid: -1.30 log S
```

The predictions align well with chemical intuition:
* **Water** (0.32) - highly soluble due to small size and polarity
* **Ethanol** (-1.59) - good solubility from hydroxyl group
* **Benzene** (-3.44) - poor solubility due to hydrophobic aromatic ring
* **Acetone** and **Acetic acid** - moderate solubility from polar functional groups

#### Understanding Model Performance

To properly evaluate our model, we need to look beyond training loss. We'll compute metrics on the test set and visualize the results:

<details>
<summary>▶ Click to see code: Model evaluation setup</summary>
<pre><code class="language-python">
# Evaluate the model on unseen molecules
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for batch in test_loader:
        pred = model(batch.x, batch.edge_index, batch.batch)
        predictions.extend(pred.squeeze().tolist())
        true_values.extend(batch.y.tolist())
</code></pre>
</details>

Now calculate standard regression metrics:

<details>
<summary>▶ Click to see code: Computing performance metrics</summary>
<pre><code class="language-python">
# Calculate standard regression metrics
rmse = np.sqrt(mean_squared_error(true_values, predictions))
r2 = r2_score(true_values, predictions)

print("\nModel Performance:")
print(f"  RMSE: {rmse:.2f} log S units")
print(f"  R² Score: {r2:.3f}")
</code></pre>
</details>

The RMSE tells us the average prediction error in log S units. The R² score indicates what fraction of the variance in solubility our model explains.

Let's visualize the predictions:

<details>
<summary>▶ Click to see code: Visualizing predictions vs truth</summary>
<pre><code class="language-python">
# Visualize prediction vs. truth
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predictions, alpha=0.5)

# Add diagonal line (perfect predictions)
min_val = min(true_values)
max_val = max(true_values)
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel('True Solubility (log S)')
plt.ylabel('Predicted Solubility (log S)')
plt.title('GNN Solubility Predictions')
plt.grid(True)
plt.show()
</code></pre>
</details>

![GNN Solubility Predictions](../../../../../resource/img/gnn/solubility.png)

The scatter plot reveals:
- Points near the diagonal indicate accurate predictions
- Vertical scatter shows prediction uncertainty
- Outliers often correspond to unusual molecular structures

#### Advanced Architecture for Better Performance

Our basic model achieved reasonable results, but we can improve performance using modern GNN techniques. Let's build a more sophisticated architecture incorporating:

1. **Residual connections** - prevent over-smoothing in deep networks
2. **Graph attention** - learn which neighbors are most important
3. **Multiple pooling strategies** - capture different molecular aspects
4. **Batch normalization** - stabilize training
5. **Dropout** - reduce overfitting

**Chemical motivation for advanced features:**
- **Attention**: Not all bonds are equally important (e.g., polar bonds matter more for solubility)
- **Residual connections**: Preserve atom identity while adding neighbor information
- **Multi-pooling**: Mean captures average properties, max captures extreme features, sum captures total content

First, let's import additional components:

<details>
<summary>▶ Click to see code: Import advanced GNN components</summary>
<pre><code class="language-python">
import torch.optim as optim
from torch_geometric.nn import GATConv, global_max_pool, global_add_pool

# GATConv: Graph Attention Network layer
# global_max_pool: Takes maximum across all atoms
# global_add_pool: Sums all atom features
</code></pre>
</details>

Now we'll expand our atom features to include more chemical information:

<details>
<summary>▶ Click to see code: Enhanced atom featurization</summary>
<pre><code class="language-python">
def atom_features_enhanced(atom):
    """Enhanced atom features including more chemical properties"""
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),     # sp, sp2, sp3
        int(atom.GetIsAromatic()),
        atom.GetMass() * 0.01,            # Scaled atomic mass
        atom.GetTotalValence(),           # Total valence
        int(atom.IsInRing()),             # Ring membership
        atom.GetTotalNumHs(),             # Hydrogen count
        int(atom.GetChiralTag() != 0)     # Chirality
    ]
</code></pre>
</details>

We also add bond features for richer edge information:

<details>
<summary>▶ Click to see code: Bond featurization</summary>
<pre><code class="language-python">
def bond_features(bond):
    """Extract bond type and properties"""
    bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4
    }
    return [
        bond_type_map.get(bond.GetBondType(), 0),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing())
    ]
</code></pre>
</details>

Now let's build the advanced architecture:

<details>
<summary>▶ Click to see code: Advanced GNN initialization</summary>
<pre><code class="language-python">
class AdvancedMolecularGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4):
        super().__init__()
        # Initial embedding
        self.embed = nn.Linear(in_dim, hidden)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch normalization
        
        for _ in range(layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        
        # Attention layer
        self.attn = GATConv(hidden, hidden, heads=4, concat=False)
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Advanced GNN prediction head</summary>
<pre><code class="language-python">
        # Prediction head with dropout
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),  # 3x for multi-pooling
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1)
        )
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Advanced GNN forward pass</summary>
<pre><code class="language-python">
    def forward(self, x, edge_index, batch):
        # Initial embedding
        x = F.relu(self.embed(x))
        
        # Graph convolutions with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = F.relu(bn(conv(x, edge_index)))
            if i > 0:  # Residual connection after first layer
                x = x + x_new
            else:
                x = x_new
        
        # Attention layer
        x = self.attn(x, edge_index)
        
        # Multi-pooling strategy
        pool1 = global_mean_pool(x, batch)
        pool2 = global_max_pool(x, batch)
        pool3 = global_add_pool(x, batch)
        
        # Concatenate different pooling results
        pooled = torch.cat([pool1, pool2, pool3], dim=1)
        
        return self.head(pooled).squeeze(-1)
</code></pre>
</details>

#### Enhanced Dataset with Richer Features

Let's create an improved dataset class that uses our enhanced features:

<details>
<summary>▶ Click to see code: Enhanced molecule dataset</summary>
<pre><code class="language-python">
class MoleculeDataset(Dataset):
    def __init__(self, smiles, targets=None):
        super().__init__()
        self._data = []
        
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
                
            mol = Chem.AddHs(mol)
            
            # Enhanced atom features
            x = torch.tensor(
                [atom_features_enhanced(a) for a in mol.GetAtoms()],
                dtype=torch.float
            )
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Enhanced edge extraction</summary>
<pre><code class="language-python">
            # Extract edges with features
            edges, edge_feats = [], []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                bf = bond_features(bond)
                
                # Add both directions
                edges.extend([[u, v], [v, u]])
                edge_feats.extend([bf, bf])
            
            # Handle single-atom molecules
            if not edges:
                edges = [[0, 0]]
                edge_feats = [[0, 0, 0]]
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Creating data object</summary>
<pre><code class="language-python">
            # Create data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([targets[i]], dtype=torch.float) if targets else None
            )
            self._data.append(data)
    
    def len(self):
        return len(self._data)
    
    def get(self, i):
        return self._data[i]
</code></pre>
</details>

#### Complete Training Pipeline

Let's put it all together with a comprehensive training loop that includes:
- Learning rate scheduling
- Early stopping
- Validation monitoring
- Performance visualization

<details>
<summary>▶ Click to see code: Data preparation</summary>
<pre><code class="language-python">
# Prepare data (subsample for faster training)
idx = np.random.choice(len(data), 500, replace=False)
smis = data.smiles.iloc[idx].tolist()
sols = data['measured log solubility in mols per litre'].iloc[idx].tolist()

# Create dataset
ds = MoleculeDataset(smis, sols)
train_ds, test_ds = torch.utils.data.random_split(ds, [400, 100])

# Create loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Model initialization and training setup</summary>
<pre><code class="language-python">
# Initialize model and training components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedMolecularGNN(in_dim=10).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
criterion = nn.MSELoss()

# Training history
train_losses = []
test_losses = []
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Training and validation loop</summary>
<pre><code class="language-python">
for epoch in range(1, 51):
    # Training phase
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        opt.step()
        
        train_loss += loss.item() * batch.num_graphs
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Validation and metrics</summary>
<pre><code class="language-python">
    # Validation phase
    model.eval()
    val_loss = 0
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            val_loss += loss.item() * batch.num_graphs
            preds.extend(out.cpu().tolist())
            trues.extend(batch.y.cpu().tolist())
    
    val_loss /= len(test_loader.dataset)
    test_losses.append(val_loss)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Print progress
    if epoch % 10 == 0:
        rmse = mean_squared_error(trues, preds, squared=False)
        r2 = r2_score(trues, preds)
        print(f"Epoch {epoch}: Train {train_loss:.3f}, "
              f"Val {val_loss:.3f}, RMSE {rmse:.3f}, R² {r2:.3f}")
</code></pre>
</details>

**Logs**
```
Epoch 10: Train 2.42, Val 3.59, RMSE 1.89, R² 0.09
Epoch 20: Train 1.87, Val 3.21, RMSE 1.79, R² 0.11
Epoch 30: Train 1.53, Val 3.05, RMSE 1.75, R² 0.12
Epoch 40: Train 1.32, Val 2.95, RMSE 1.72, R² 0.12
Epoch 50: Train 1.18, Val 2.87, RMSE 1.69, R² 0.12
```

#### Visualization and Analysis

Finally, let's create comprehensive visualizations to understand our model's performance:

<details>
<summary>▶ Click to see code: Performance visualization</summary>
<pre><code class="language-python">
# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Training curves
ax1.plot(train_losses, label='Train')
ax1.plot(test_losses, label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training Progress')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Predictions vs Truth
ax2.scatter(trues, preds, alpha=0.6)
ax2.plot([min(trues), max(trues)], [min(trues), max(trues)], 'r--')
ax2.set_xlabel('True Solubility (log S)')
ax2.set_ylabel('Predicted Solubility (log S)')
ax2.set_title(f'Predictions (R² = {r2:.2f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
</code></pre>
</details>

![Training Progress and Predictions](../../../../../resource/img/gnn/training_and_predictions.png)

#### Final Predictions on New Molecules

Let's create a user-friendly prediction function:

<details>
<summary>▶ Click to see code: Enhanced prediction function</summary>
<pre><code class="language-python">
def predict_solubility_enhanced(smi, model, device):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None
    
    # Prepare features
    mol = Chem.AddHs(mol)
    x = torch.tensor(
        [atom_features_enhanced(a) for a in mol.GetAtoms()],
        dtype=torch.float
    ).to(device)
    
    # Prepare edges
    edges = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[u, v], [v, u]])
    
    if not edges:
        edges = [[0, 0]]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        return model(x, edge_index, batch).item()
</code></pre>
</details>

<details>
<summary>▶ Click to see code: Testing on diverse molecules</summary>
<pre><code class="language-python">
# Test on chemically diverse molecules
test_mols = [
    ("O", "Water"),
    ("CCO", "Ethanol"),
    ("c1ccccc1", "Benzene"),
    ("CC(=O)O", "Acetic acid"),
    ("CC(C)CC(C)(C)C", "Octane"),
    ("O=C(O)c1ccccc1O", "Salicylic acid")
]

print("\nPredictions for test molecules:")
print("-" * 40)
for smi, name in test_mols:
    pred = predict_solubility_enhanced(smi, model, device)
    print(f"{name:<15} {smi:<20} → {pred:>6.2f} log S")
</code></pre>
</details>

**Results**
```
Predictions for test molecules:
----------------------------------------
Water           O                    →   0.28 log S
Ethanol         CCO                  →  -1.45 log S
Benzene         c1ccccc1             →  -3.12 log S
Acetic acid     CC(=O)O              →  -1.20 log S
Octane          CC(C)CC(C)(C)C       →  -5.83 log S
Salicylic acid  O=C(O)c1ccccc1O      →  -2.74 log S
```

#### Key Takeaways

1. **Data Understanding**: Always explore and visualize your molecular data before modeling
2. **Rich Featurization**: Include diverse chemical features (hybridization, aromaticity, chirality)
3. **Architecture Matters**: Residual connections, attention, and multi-pooling improve performance
4. **Careful Evaluation**: Monitor both training and validation loss to detect overfitting
5. **Chemical Validation**: Check that predictions align with chemical intuition

This complete pipeline demonstrates how GNNs can learn meaningful structure-property relationships directly from molecular graphs. The same framework easily adapts to other properties - simply change the target values and the model will discover the relevant patterns.

### 3.3.4 Challenges and Interpretability in GNNs

#### Completed and Compiled Code: [Click Here](https://colab.research.google.com/drive/1rtJ6voxMVK7KS-oZS97_7Jr1bBf7Z15T?usp=sharing)

While GNNs have shown remarkable success in molecular property prediction, they face several fundamental challenges that limit their practical deployment. In this section, we'll explore two critical issues: the over-smoothing phenomenon that limits network depth, and the interpretability challenge that makes it difficult to understand model predictions.

#### The Power of Depth vs. The Curse of Over-smoothing

In Graph Neural Networks (GNNs), adding more message-passing layers allows nodes (atoms) to gather information from increasingly distant parts of a graph (molecule). At first glance, it seems deeper networks should always perform better—after all, more layers mean more context. But in practice, there's a major trade-off known as **over-smoothing**.

**What to Demonstrate**

Before we jump into the code, here's **what it's trying to show**:

We want to measure how **similar node embeddings become** as we increase the number of GCN layers. If all node vectors become nearly identical after several layers, that means the model is **losing resolution**—different atoms can't be distinguished anymore. This is called **over-smoothing**.

**Functions and Concepts Used**

* **`GCNConv` (from `torch_geometric.nn`)**: This is a standard Graph Convolutional Network (GCN) layer. It performs message passing by aggregating neighbor features and updating node embeddings. It normalizes messages by node degrees to prevent high-degree nodes from dominating.

* **`F.relu()`**: Applies a non-linear ReLU activation function after each GCN layer. This introduces non-linearity to the model, allowing it to learn more complex patterns.

* **`F.normalize(..., p=2, dim=1)`**: This normalizes node embeddings to unit length (L2 norm), which is required for cosine similarity calculation.

* **`torch.mm()`**: Matrix multiplication is used here to compute the full cosine similarity matrix between normalized node embeddings.

* **Cosine similarity**: Measures how aligned two vectors are (value close to 1 means very similar). By averaging all pairwise cosine similarities, we can track whether the node representations are collapsing into the same vector.

**Graph Construction**

We use a **6-node ring structure** as a simple molecular graph. Each node starts with a unique identity (using identity matrix `torch.eye(6)` as input features), and all nodes are connected in a cycle:

<details>
<summary>▶ Click to see code: Constructing a simple cyclic graph</summary>
<pre><code class="language-python">
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
</code></pre>
</details>

**Over-smoothing Analysis**

Now we apply the same GCN layer multiple times to simulate a deeper GNN. After each layer, we re-compute the node embeddings and compare them using cosine similarity:

<details>
<summary>▶ Click to see code: Demonstrating over-smoothing</summary>
<pre><code class="language-python">
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
</code></pre>
</details>

**Output**
```
Depth 1: Average similarity = 0.406
Depth 3: Average similarity = 0.995
Depth 5: Average similarity = 0.993
Depth 10: Average similarity = 1.000
```

![Over-smoothing in GNNs](../../resource/img/gnn/oversmoothing.png)

*As shown above, as the number of message-passing layers increases, node representations converge. Initially distinct feature vectors (left) become nearly indistinguishable after several layers (right), resulting in the loss of structural information. This phenomenon is known as **over-smoothing** and is a critical limitation of deep GNNs.*

**Interpretation**

As we can see, even at just 3 layers, the node embeddings become nearly identical. By 10 layers, the model has effectively lost all ability to distinguish individual atoms. This is the core issue of **over-smoothing**—deep GNNs can blur out meaningful structural differences.

To mitigate this problem, modern GNNs use techniques like:
* **Residual connections** (skip connections that reintroduce raw input)
* **Feature concatenation from earlier layers**
* **Batch normalization or graph normalization**
* **Jumping knowledge networks** to combine representations from multiple layers

When working with molecular graphs, you should **choose the depth of your GNN carefully**. It should be **deep enough** to capture important substructures, but **not so deep** that you lose atomic-level details.

#### Interpretability in Molecular GNNs

Beyond the technical challenge of over-smoothing, GNNs face a critical issue of interpretability. When a model predicts that a molecule might be toxic or have specific properties, chemists need to understand which structural features drive that prediction. This "black box" nature of neural networks is particularly problematic in chemistry, where understanding structure-activity relationships is fundamental to rational drug design.

Recent advances in GNN interpretability for molecular applications have taken several promising directions:

**Attention-Based Methods**: Graph Attention Networks (GATs) provide built-in interpretability through their attention mechanisms, allowing researchers to visualize which atoms or bonds the model considers most important for a given prediction [1,2]. This approach naturally aligns with chemical intuition about reactive sites and functional groups.

**Substructure-Based Explanations**: The Substructure Mask Explanation (SME) method represents a significant advance by providing interpretations based on chemically meaningful molecular fragments rather than individual atoms or edges [3]. This approach uses established molecular segmentation methods to ensure explanations align with chemists' understanding, making it particularly valuable for identifying pharmacophores and toxicophores.

**Integration of Chemical Knowledge**: Recent work has shown that incorporating pharmacophore information hierarchically into GNN architectures not only improves prediction performance but also enhances interpretability by explicitly modeling chemically meaningful substructures [4]. This bridges the gap between data-driven learning and domain expertise.

**Gradient-Based Attribution**: Methods like SHAP (SHapley Additive exPlanations) have been successfully applied to molecular property prediction, providing feature importance scores that help identify which molecular characteristics most influence predictions [5,6]. These approaches are particularly useful for understanding global model behavior across different molecular classes.

**Comparative Studies**: Recent comparative studies have shown that while GNNs excel at learning complex patterns, traditional descriptor-based models often provide better interpretability through established chemical features, suggesting a potential hybrid approach combining both paradigms [6].

The field is moving toward interpretable-by-design architectures rather than post-hoc explanation methods. As noted by researchers, some medicinal chemists value interpretability over raw accuracy if a small sacrifice in performance can significantly enhance understanding of the model's reasoning [3]. This reflects a broader trend in molecular AI toward building systems that augment rather than replace human chemical intuition.

**References:**

[1] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. *International Conference on Learning Representations*.

[2] Yuan, H., Yu, H., Gui, S., & Ji, S. (2022). Explainability in graph neural networks: A taxonomic survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

[3] Chemistry-intuitive explanation of graph neural networks for molecular property prediction with substructure masking. (2023). *Nature Communications*, 14, 2585.

[4] Integrating concept of pharmacophore with graph neural networks for chemical property prediction and interpretation. (2022). *Journal of Cheminformatics*, 14, 49.

[5] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

[6] Jiang, D., Wu, Z., Hsieh, C. Y., Chen, G., Liao, B., Wang, Z., ... & Hou, T. (2021). Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models. *Journal of Cheminformatics*, 13(1), 1-23.

#### Summary

The challenges facing molecular GNNs—over-smoothing and interpretability—are significant but surmountable. Over-smoothing limits the depth of networks we can effectively use, constraining the model's ability to capture long-range molecular interactions. Meanwhile, the interpretability challenge affects trust and adoption in real-world applications where understanding model decisions is crucial.

Current solutions include architectural innovations like residual connections to combat over-smoothing, and various interpretability methods ranging from attention visualization to substructure-based explanations. The key insight is that effective molecular AI systems must balance predictive power with chemical interpretability, ensuring that models not only make accurate predictions but also provide insights that align with and enhance human understanding of chemistry.

As the field progresses, the focus is shifting from purely accuracy-driven models to systems that provide transparent, chemically meaningful explanations for their predictions. This evolution is essential for GNNs to fulfill their promise as tools for accelerating molecular discovery and understanding.

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
<summary>▶ Show Solution Code</summary>

```python
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
```

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
<summary>▶ Show Solution Code</summary>

```python
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
```

</details>
