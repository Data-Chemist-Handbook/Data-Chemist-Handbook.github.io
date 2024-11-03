# **3.1.1 Decision Trees** #

**Decision Trees** are intuitive and powerful models used in machine
learning to make predictions and decisions. Think of it like playing a
game of 20 questions, where each question helps you narrow down the
possibilities. Decision trees function similarly; they break down a
complex decision into a series of simpler questions based on the
data.

Each question, referred to as a \"decision,\" relies on a specific
characteristic or feature of the data. For instance, if you\'re trying
to determine whether a fruit is an apple or an orange, the initial
question might be, \"Is the fruit\'s color red or orange?\" Depending on
the answer, you might follow up with another question---such as, \"Is
the fruit\'s size small or large?\" This questioning process continues
until you narrow it down to a final answer (e.g., the fruit is either an
apple or an orange).

In a decision tree, these questions are represented as nodes, and the
possible answers lead to different branches. The final outcomes are
represented at the end of each branch, known as leaf nodes. One of the
key advantages of decision trees is their clarity and ease of
understanding---much like a flowchart. However, they can also be prone
to overfitting, especially when dealing with complex datasets that have
many features. Overfitting occurs when a model performs exceptionally
well on training data but fails to generalize to new or unseen
data.

In summary, decision trees offer an intuitive approach to making
predictions and decisions, but caution is required to prevent them from
becoming overly complicated and tailored too closely to the training
data.

# **3.1.2 Random Forest** #

**Random Forests** address the limitations of decision trees by
utilizing an ensemble of multiple trees instead of relying on a single
one. Imagine you're gathering opinions about a game outcome from a group
of people; rather than trusting just one person\'s guess, you ask
everyone and then take the most common answer. This is the essence of
how a Random Forest operates.

In a Random Forest, numerous decision trees are constructed, each
making its own predictions. However, a key difference is that each tree
is built using a different subset of the data and considers different
features of the data. This technique, known as bagging (Bootstrap
Aggregating), allows each tree to provide a unique perspective, which
collectively leads to a more reliable prediction.

When making a final prediction, the Random Forest aggregates the
predictions from all the trees. For classification tasks, it employs
majority voting to determine the final class label, while for regression
tasks, it averages the results.

Random Forests typically outperform individual decision trees because
they are less likely to overfit the data. By combining multiple trees,
they achieve a balance between model complexity and predictive
performance on unseen data.

#### **Real-Life Analogy**

Consider Andrew, who wants to decide on a destination for his year-long
vacation. He starts by asking his close friends for suggestions. The
first friend asks Andrew about his past travel preferences, using his
answers to recommend a destination. This is akin to a decision tree
approach---one friend following a rule-based decision process.

Next, Andrew consults more friends, each of whom poses different
questions to gather recommendations. Finally, Andrew chooses the places
suggested most frequently by his friends, mirroring the Random Forest
algorithm\'s method of aggregating multiple decision trees\'
outputs.

### **Implementing Random Forest**

The following code demonstrates how to implement a simple Random Forest
algorithm in Python using the QSAR dataset. The QSAR (Quantitative
Structure-Activity Relationship) dataset is primarily utilized in
cheminformatics and computational biology to predict the activity or
properties of chemical compounds based on their molecular structures. It
contains 41 predictor columns in the CSV file.

**Example code:**

<pre>
    <code class="python">
!pip install scikit-learn # install needed library

# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.ensemble import RandomForestClassifier  # For building a Random Forest model
from sklearn.metrics import accuracy_score  # For measuring model accuracy

# Load the dataset from a CSV file
data = pd.read_csv('qsar-biodeg.csv')  # Assumes the file 'qsar-biodeg.csv' is in the same directory

# Separate the independent variables (features) and the target variable (label)
X = data.iloc[:, :-1]  # All columns except the last one as features
y = data.iloc[:, -1]   # The last column as the target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier with 100 trees and a fixed random state for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model using the training data
rf_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the accuracy of the model by comparing the predictions to the actual test labels
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the model, formatted to two decimal places
print(f"Model accuracy: {accuracy * 100:.2f}%")
    </code>
</pre>

With this model, we achieved an accuracy score of **90.05%**.

In this Random Forest classification model, adjusting specific parameters can significantly impact performance and efficiency. The test_size parameter in the train_test_split function determines the proportion of data reserved for testing. A larger test_size provides more data to evaluate model accuracy but reduces the data available for training, potentially affecting accuracy. The n_estimators parameter in RandomForestClassifier controls the number of trees in the forest. Increasing the number of trees typically enhances accuracy by reducing variance, though it requires more computational resources, while a lower n_estimators value can speed up training at the cost of some accuracy. Finally, the random_state parameter sets a seed to ensure reproducible results. While it doesn't affect model performance directly, it guarantees consistent results across runs, which can be helpful in iterative model refinement. Adjusting these parameters thoughtfully can help achieve a balance between accuracy and efficiency.

# Approaching Random Forest Problems

When tackling a classification or regression problem using the Random Forest algorithm, a systematic approach can enhance your chances of success. Here’s a step-by-step guide to effectively solve any Random Forest problem:

1. **Understand the Problem Domain**: Begin by thoroughly understanding the problem you are addressing. Identify the nature of the data and the specific goal—whether it's classification (e.g., predicting categories) or regression (e.g., predicting continuous values). Familiarize yourself with the dataset, including the features (independent variables) and the target variable (dependent variable).

2. **Data Collection and Preprocessing**: Gather the relevant dataset and perform necessary preprocessing steps. This may include handling missing values, encoding categorical variables, normalizing or standardizing numerical features, and removing any outliers. Proper data cleaning ensures that the model learns from quality data.

3. **Exploratory Data Analysis (EDA)**: Conduct an exploratory data analysis to understand the underlying patterns, distributions, and relationships within the data. Visualizations, such as scatter plots, histograms, and correlation matrices, can provide insights that inform feature selection and model tuning.

4. **Feature Selection and Engineering**: Identify the most relevant features for the model. This can be achieved through domain knowledge, statistical tests, or feature importance metrics from preliminary models. Consider creating new features through feature engineering to enhance model performance.

5. **Model Training and Parameter Tuning**: Split the dataset into training and testing sets, typically using an 80-20 or 70-30 ratio. Train the Random Forest model using the training data, adjusting parameters such as the number of trees (`n_estimators`), the maximum depth of the trees (`max_depth`), and the minimum number of samples required to split an internal node (`min_samples_split`). Utilize techniques like grid search or random search to find the optimal hyperparameters.

6. **Model Evaluation**: Once trained, evaluate the model's performance on the test set using appropriate metrics. For classification problems, metrics such as accuracy, precision, recall, F1 score, and ROC-AUC are valuable. For regression tasks, consider metrics like mean absolute error (MAE), mean squared error (MSE), and R-squared.

7. **Interpretation and Insights**: Analyze the model’s predictions and feature importance to derive actionable insights. Understanding which features contribute most to the model can guide decision-making and further improvements in the model or data collection.

8. **Iterate and Improve**: Based on the evaluation results, revisit the previous steps to refine your model. This may involve further feature engineering, collecting more data, or experimenting with different algorithms alongside Random Forest to compare performance.

9. **Deployment**: Once satisfied with the model's performance, prepare it for deployment. Ensure the model can process incoming data and make predictions in a real-world setting, and consider implementing monitoring tools to track its performance over time.

By following this structured approach, practitioners can effectively leverage the Random Forest algorithm to solve a wide variety of problems, ensuring thorough analysis, accurate predictions, and actionable insights.

## Practice Problems ##

Now it's your turn! Below are Five Random Forest Problems, as well as corresponding datasets. Try your hand at solving them!

# Practice Problems Using Random Forest

## Problem 1: Iris Species Classification
- **Dataset**: Iris dataset. To load this dataset, execute the following command in your python script: 
<pre>
    <code class="python">
from sklearn import datasets


iris = datasets.load_iris()
    </code>
</pre>
The iris dataset will now be stored in the iris variable. Feel free to print it out to get an understanding of what it contains
- **Task**: Use the Random Forest algorithm to classify different species of iris flowers based on features such as sepal length, sepal width, petal length, and petal width.
- **Goal**: Predict the species (Setosa, Versicolor, or Virginica) based on the provided measurements.

## Problem 2: Wine Quality Prediction
- **Dataset**: [Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Task**: Implement a Random Forest model to predict the quality of red and white wine based on physicochemical tests (e.g., acidity, sugar level, pH).
- **Goal**: Predict the quality score (ranging from 0 to 10) and evaluate the model’s accuracy.

## Problem 3: Titanic Survival Prediction
- **Dataset**: [Titanic dataset](https://www.kaggle.com/c/titanic/data)
- **Task**: Use the Random Forest algorithm to predict whether a passenger survived the Titanic disaster based on features such as age, gender, class, and fare.
- **Goal**: Predict survival (1 for survived, 0 for did not survive) and analyze feature importance.

## Problem 4: Heart Disease Diagnosis
- **Dataset**: [Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Task**: Build a Random Forest classifier to determine the presence of heart disease based on clinical attributes such as age, cholesterol levels, blood pressure, and other health metrics.
- **Goal**: Classify patients as having heart disease (1) or not (0) and assess the model's performance.

## Problem 5: Handwritten Digit Recognition
- **Dataset**: [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- **Task**: Implement a Random Forest classifier to recognize handwritten digits (0-9) based on pixel intensity values.
- **Goal**: Classify images of handwritten digits and evaluate the model’s accuracy on a test set.



### **Strengths and Weaknesses of Random Forest**

**Strengths:**

-   **Robustness**: Random Forests are less prone to overfitting
    compared to individual decision trees, making them more reliable for
    new data.

-   **Versatility**: They can handle both classification and regression
    tasks effectively.

-   **Feature Importance**: Random Forests provide insights into the
    significance of each feature in making predictions.

**Weaknesses:**

-   **Complexity**: The model can become complex, making it less
    interpretable than single decision trees.

-   **Resource Intensive**: Training a large number of trees can
    require significant computational resources and time.

-   **Slower Predictions**: While individual trees are quick to
    predict, aggregating predictions from multiple trees can slow down
    the prediction process.

[[https://mlu-explain.github.io/random-forest/]{.underline}](https://mlu-explain.github.io/random-forest/)
this has really good diagrams but not exactly sure how to incorporate
them