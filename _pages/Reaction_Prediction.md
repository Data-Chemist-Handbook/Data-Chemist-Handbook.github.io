---
title: 6. Reaction Prediction
author: Haomin
date: 2024-08-16
category: Jekyll
layout: post
---

dataset: USPTO subset (https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00064)

## 6.1 RNNs and LSTM

- **Recurrent Neural Networks (RNNs)**
  - RNNs are designed to process sequential data, making them ideal for tasks that depend on time-series or ordered inputs.
  - Common applications include:
    - Time series forecasting (e.g., predicting flood levels).
    - Natural language processing (e.g., language translation, sentiment analysis).
    - Speech recognition.
    - Image captioning.

- **Key Characteristics of RNNs**
  - "Memory" function that uses past inputs to influence current output.
  - Different from feedforward networks due to their ability to handle sequences.
  - Uses shared parameters across network layers, unlike feedforward networks which have unique weights for each node.

- **RNN Operations**
  - Processes input step-by-step while maintaining a hidden state that holds information from prior steps.
  - Utilizes backpropagation through time (BPTT) to calculate gradients and update weights.

- **Limitations of Traditional RNNs**
  - Struggles with long-term dependencies due to vanishing gradient problems.

- **LSTM Networks**
  - A specialized type of RNN designed to capture long-term dependencies.
  - Uses gates (input, forget, and output) to regulate the flow of information.

- **Use Cases for LSTM**
  - Time series analysis for experimental data.
  - Prediction of chemical properties.
  - Simulation of reactions over time.
## Python Code Snippet for Reaction Prediction Using RNNs and LSTMs

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load and preprocess your dataset (e.g., BACE Scaffold dataset from https://paperswithcode.com/dataset/bace-scaffold)
# Example: Assume input_data and target_data are preprocessed and ready for training

# Define the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))  # Example embedding layer
model.add(LSTM(128))  # LSTM layer with 128 units
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example input data (replace with preprocessed chemical sequences)
input_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Replace with actual sequence data
target_data = [0, 1, 0]  # Replace with actual target labels

# Train the model
model.fit(input_data, target_data, epochs=10)

# Make predictions
predictions = model.predict(input_data)
```
### How the Code Works
- **Importing Required Libraries**:
  - `tensorflow` for building neural networks.
  - `Sequential`, `Embedding`, `LSTM`, and `Dense` layers from `tensorflow.keras` for constructing the model.

- **Defining the Model**:
  - Create a `Sequential` model, which means layers are added one after another.
  - Add an `Embedding` layer that helps the model understand the input features.
  - Include an `LSTM` layer with 128 units to capture dependencies in the input data.
  - Add a `Dense` layer with a `sigmoid` activation function for binary classification.

- **Compiling the Model**:
  - Use the `Adam` optimizer to adjust the learning rate during training.
  - Use `binary_crossentropy` as the loss function since this example involves binary classification.
  - Track accuracy as a metric.

- **Preparing Data**:
  - Replace `input_data` and `target_data` with data from the BACE Scaffold dataset after preprocessing.
  - Ensure the dataset is formatted as sequences of numbers (e.g., molecular features) for the model.

- **Training the Model**:
  - Train using `model.fit()` by providing input data and corresponding target outputs.
  - Use `epochs=10` to iterate the training process multiple times to improve learning.

- **Making Predictions**:
  - Use `model.predict()` to make predictions based on new input data.
  - This output can help indicate the likelihood of a specific chemical reaction occurring.

- **Timesteps & Features**: Ensure your input shape matches the data format, where `timesteps` refer to the number of time points and `features` refer to properties recorded at each time point.
- **Data Example**: Sequences should represent input variables like time-dependent chemical properties (e.g., temperature, concentration).

