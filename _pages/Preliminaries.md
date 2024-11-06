---
title: 2. Preliminaries
author: Haomin, Dan Yoo
date: 2024-11-06
category: Jekyll
layout: post
---

In the Preliminaries character, we will introduce some base coding still and data analysis still for the beginners.

## 2.1 Introduction to Python

This section covers essential Python skills, including setting up your environment, understanding basic syntax, and using packages to aid data analysis. This foundational knowledge is valuable for beginners and will support more advanced data analysis in later sections.

### 2.1.1 Setting up Python Environment

To get started with Python, we’ll set up a development environment using **Anaconda** and **Jupyter Notebook**.

- **Anaconda**: A package manager and environment manager commonly used for data science. It simplifies package installation and management.
- **Jupyter Notebook**: An interactive environment ideal for data exploration and analysis. Jupyter Notebooks can be launched directly from Anaconda.

#### Steps to Set Up:

1. [Download Anaconda](https://www.anaconda.com/products/distribution) and install it according to your operating system.
2. Launch Anaconda Navigator and start **Jupyter Notebook**.
3. Once Jupyter Notebook opens in your browser, create a new notebook and start coding!

### 2.1.2 Basic Syntax and Structure

Python's simple syntax makes it a powerful and beginner-friendly language for data analysis. Here, we’ll cover core aspects:

#### Variables, Loops, and Functions

**Variables**: Used to store data. You can define a variable by simply assigning it a value.

<pre>
    <code class="python">
# Defining variables
compound_name = "Aspirin"
molecular_weight = 180.16
    </code>
</pre>

**Loops**: Used to perform repetitive tasks.

<pre>
    <code class="python">
# For loop example
for i in range(3):
    print(f"Compound {i+1}")
    </code>
</pre>

**Functions**: Functions in Python allow you to reuse blocks of code and organize your script.

<pre>
    <code class="python">
# Function to calculate the molecular weight ratio
def molecular_weight_ratio(compound_weight, standard_weight=100):
    return compound_weight / standard_weight

print(molecular_weight_ratio(molecular_weight))
    </code>
</pre>

### 2.1.3 Python Packages

Python packages are pre-built libraries that simplify data analysis. Here, we’ll focus on a few essential packages for our work.

#### Key Packages

1. **NumPy**: Used for numerical computing, especially helpful for handling arrays and performing mathematical operations.
2. **Pandas**: A popular library for data manipulation, ideal for handling tabular data structures.
3. **Matplotlib** and **Seaborn**: Libraries for data visualization.

#### Example Code to Install and Import Packages

<pre>
    <code class="python">
# Installing packages
!pip install numpy pandas matplotlib seaborn

# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    </code>
</pre>

#### Working with JSON Data

**JSON (JavaScript Object Notation)** is a common data format for sharing data between systems, especially in APIs. JSON files are lightweight and easy to parse.

<pre>
    <code class="python">
import json

# Loading data from a JSON file
with open('data.json') as f:
    data = json.load(f)

# Accessing data in JSON format
print(data['compound_name'])
    </code>
</pre>

### Practice Problem

**Problem**: Write Python code to create a variable for a compound’s molecular weight, set it to 180.16, and then create a function that doubles the molecular weight.

**Solution**

<pre>
    <code class="python">
# Define a variable for molecular weight
molecular_weight = 180.16

# Function to double the molecular weight
def double_weight(weight):
    return weight * 2

# Test the function
print(f"Double molecular weight: {double_weight(molecular_weight)}")
    </code>
</pre>

## 2.2 Data Analysis with Python

In this chapter, we'll explore how to use Python for data analysis, focusing on importing and managing datasets commonly encountered in chemistry. Data analysis is a crucial skill for chemists, allowing you to extract meaningful insights from experimental data, predict outcomes, and make informed decisions in your research.

### 2.2.1 Importing and Managing Datasets

Effective data analysis begins with properly importing and managing your datasets. This section will guide you through loading data from various file formats, including those specific to chemistry, and handling data from databases.

#### 2.2.1.1 Loading Data from Various File Formats

**Reading Data from CSV, Excel, and JSON Files**

**Explanation:**

CSV (Comma-Separated Values), Excel, and JSON (JavaScript Object Notation) are common file formats for storing tabular data. Python's `pandas` library provides straightforward methods to read these files into DataFrames, which are powerful data structures for data manipulation.

**Example Code:**

<pre>
    <code class="python">
import pandas as pd

# Reading a CSV file
csv_data = pd.read_csv('experimental_data.csv')

# Reading an Excel file
excel_data = pd.read_excel('compound_properties.xlsx', sheet_name='Sheet1')

# Reading a JSON file
json_data = pd.read_json('reaction_conditions.json')
    </code>
</pre>

**Explanation of the Code:**

- `pd.read_csv()` reads data from a CSV file into a DataFrame.
- `pd.read_excel()` reads data from an Excel file. The `sheet_name` parameter specifies which sheet to read.
- `pd.read_json()` reads data from a JSON file.

**Practice Problem:**

You have been provided with a CSV file named BBBP.csv, which contains information about various compounds and their blood-brain barrier permeability. Write Python code to:

Read the CSV file into a DataFrame using pd.read_csv().
Display the first five rows of the DataFrame using df.head().
Calculate the proportion of permeable compounds, i.e., those for which the p_np column is 1.


**Answer:**
<pre> 
    <code class="python"> 
    import pandas as pd 
    # 1. Read the CSV file into a DataFrame 
    df = pd.read_csv('BBBP.csv') 
    # 2. Display the first five rows 
    print(df.head()) 
    # 3. Calculate the proportion of permeable compounds 
    permeable_ratio = df['p_np'].mean() 
    print(f"Proportion of permeable compounds: {permeable_ratio:.2f}") 
    </code> 
</pre>

### 2.2.1.2 Data Cleaning and Preprocessing

**Handling Missing Values and Duplicates**

**Explanation:**
Data cleaning involves dealing with missing or incorrect data entries to improve the quality of the dataset. Handling missing values and removing duplicates ensures that analyses are accurate and reliable.

**Example Code:**

<pre>
    <code class="python">
import pandas as pd

# Loading the BBBP dataset
df = pd.read_csv('BBBP.csv')

# Handling missing values: fill missing names with 'Unknown' and smiles with an empty string
df_filled = df.fillna({'name': 'Unknown', 'smiles': ''})

# Removing duplicate rows based on the SMILES column
df_no_duplicates = df.drop_duplicates(subset=['smiles'])

# Displaying the first five rows of the cleaned dataset
print(df_filled.head())

# Displaying the number of rows after removing duplicates
print(f"Number of rows after removing duplicates: {len(df_no_duplicates)}")
    </code>
</pre>

- `fillna()` is used to fill missing values with specified values.
- `drop_duplicates()` removes duplicate rows from the DataFrame.

**Practice Problem:**
We will clean the dataset by filling missing `name` and `smiles` values and removing any duplicate entries based on smiles.

Given a DataFrame with missing values:
1. Fill missing values in the `name` column with `'Unknown'` and in the `smiles` column with an `empty string`.
2. Remove any duplicate rows based on the `smiles` column.
---

**Data Type Conversions**

**Explanation:**
Converting data types ensures consistency and prevents errors, especially when performing mathematical calculations or comparisons. This is necessary when data is imported with incorrect types (e.g., numbers stored as strings).

**Example Code:**

<pre>
    <code class="python">
import pandas as pd

# Example DataFrame with mixed types
data = {'Compound': ['A', 'B', 'C'],
        'Quantity': ['10', '20', '30'],
        'Purity': [99.5, 98.7, 97.8]}
df = pd.DataFrame(data)

# Converting 'Quantity' to integer
df['Quantity'] = df['Quantity'].astype(int)

# Converting 'Purity' to string
df['Purity'] = df['Purity'].astype(str)

print(df.dtypes)
    </code>
</pre>

**Practice Problem:**
In the BBBP dataset, the `num` column (compound number) should be treated as an integer, and the `p_np` column (permeability label) should be converted to categorical data.
1. Convert the num column to integer and the p_np column to a categorical type.
2. Verify that the conversions are successful by printing the data types.

**Solution**
<pre>
    <code>
        import pandas as pd

        # Loading the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Convert 'num' to integer and 'p_np' to categorical
        df['num'] = df['num'].astype(int)
        df['p_np'] = df['p_np'].astype('category')

        # Print the data types of the columns
        print(df.dtypes)
    </code>
</pre>
---

**Normalizing and Scaling Data**

**Explanation:**
Normalization adjusts the values of numerical columns to a common scale without distorting differences in ranges. This is often used in machine learning algorithms to improve model performance by making data more comparable.

**Example Code:**

<pre>
    <code class="python">
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Example DataFrame
data = {'Compound': ['A', 'B', 'C'],
        'Concentration': [0.1, 0.3, 0.5],
        'pH': [7.0, 6.5, 8.0]}
df = pd.DataFrame(data)

# Normalizing the 'Concentration' and 'pH' columns
scaler = MinMaxScaler()
df[['Concentration', 'pH']] = scaler.fit_transform(df[['Concentration', 'pH']])

print(df)
    </code>
</pre>

**Practice Problem:**
We’ll normalize the `num` column using Min-Max scaling, which adjusts values to a common scale between 0 and 1.
1. Normalize the num column in the BBBP dataset using Min-Max scaling.
2. Print the first few rows to verify the normalization.

**Solution**
<pre>
    <code>
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        # Loading the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Normalizing the 'num' column (compound number)
        scaler = MinMaxScaler()
        df[['num']] = scaler.fit_transform(df[['num']])

        # Print the first few rows of the normalized dataset
        print(df.head())
    </code>
</pre>
---

**Encoding Categorical Variables**

**Explanation:**

Encoding converts non-numeric data (like categories) into numeric values so that machine learning models can process them. Common encoding methods include one-hot encoding and label encoding.

**Example Code:**

<pre>
    <code class="python">
import pandas as pd

# Example DataFrame with categorical data
data = {'Compound': ['A', 'B', 'C'],
        'State': ['solid', 'liquid', 'gas']}
df = pd.DataFrame(data)

# One-hot encoding the 'State' column
df_encoded = pd.get_dummies(df, columns=['State'])

print(df_encoded)
    </code>
</pre>

**Practice Problem:**
The `p_np` column is a binary classification of permeability (1 or 0). We will apply one-hot encoding to this column.
1. Apply one-hot encoding to the p_np column in the BBBP dataset.
2. Print the first few rows of the encoded DataFrame to verify the result.

**Solution**
<pre>
    <code>
        import pandas as pd

        # Loading the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Apply one-hot encoding to the 'p_np' column
        df_encoded = pd.get_dummies(df, columns=['p_np'], prefix='permeability')

        # Print the first few rows of the encoded DataFrame
        print(df_encoded.head())
    </code>
</pre>
---
### 2.2.1.3 Data Manipulation with Pandas

**Filtering and Selecting Data**

**Explanation:**

Filtering allows you to select specific rows or columns from a DataFrame that meet a certain condition. This is useful for narrowing down data to relevant sections.

**Example Code:**

<pre>
    <code class="python">
        import pandas as pd

        # Example DataFrame
        data = {'Compound': ['A', 'B', 'C'],
                'MolecularWeight': [180.16, 250.23, 320.45]}
        df = pd.DataFrame(data)

        # Filtering rows where MolecularWeight is greater than 200
        filtered_df = df[df['MolecularWeight'] > 200]

        print(filtered_df)
    </code>
</pre>

**Practice Problem:**

1. Filter a DataFrame from the BBBP dataset to show only rows where the `num` (compound number) is greater than 500.
2. Select a subset of columns from the dataset and display only the `name` and `smiles` columns.

**Solution**
<pre>
    <code class="python">
        import pandas as pd

        # Loading the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Filtering rows where the 'num' column is greater than 500
        filtered_df = df[df['num'] > 500]

        # Selecting a subset of columns: 'name' and 'smiles'
        subset_df = df[['name', 'smiles']]

        print(filtered_df.head())
        print(subset_df.head())
    </code>
</pre>

---

**Merging and Joining Datasets**

**Explanation:**

Merging allows for combining data from multiple DataFrames based on a common column or index. This is especially useful for enriching datasets with additional information.

**Example Code:**

<pre>
    <code class="python">
        import pandas as pd

        # Example DataFrames
        df1 = pd.DataFrame({'Compound': ['A', 'B'],
                            'MolecularWeight': [180.16, 250.23]})

        df2 = pd.DataFrame({'Compound': ['A', 'B'],
                            'MeltingPoint': [120, 150]})

        # Merging DataFrames on the 'Compound' column
        merged_df = pd.merge(df1, df2, on='Compound')

        print(merged_df)
    </code>
</pre>

**Practice Problem:**

1. Merge two DataFrames from the BBBP dataset: One containing the `name` and `smiles` columns and another containing the `num` and `p_np` columns.
2. Perform a left join on the `name` column and display the result.

**Solution**
<pre>
    <code class="python">
        import pandas as pd

        # Loading the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Create two DataFrames
        df1 = df[['name', 'smiles']]
        df2 = df[['name', 'num', 'p_np']]

        # Perform a left join on the 'name' column
        merged_df = pd.merge(df1, df2, on='name', how='left')

        # Print the merged DataFrame
        print(merged_df.head())
    </code>
</pre>

---

**Grouping and Aggregation**

**Explanation:**

Grouping organizes data based on specific columns, and aggregation provides summary statistics like the sum, mean, or count. This is useful for analyzing data at a higher level.

**Example Code:**

<pre>
    <code class="python">
        import pandas as pd

        # Example DataFrame
        data = {'Compound': ['A', 'A', 'B', 'B'],
                'Measurement': [1, 2, 3, 4]}
        df = pd.DataFrame(data)

        # Grouping by 'Compound' and calculating the sum
        grouped_df = df.groupby('Compound').sum()

        print(grouped_df)
    </code>
</pre>

**Practice Problem:**

1. Group the BBBP dataset by `p_np` and compute the average `num` for each group (permeable and non-permeable compounds).
2. Use multiple aggregation functions (e.g., count and mean) on the `num` column.

**Solution**
<pre>
    <code class="python">
        import pandas as pd

        # Loading the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Grouping by 'p_np' and calculating the average 'num'
        grouped_df = df.groupby('p_np')['num'].mean()

        # Applying multiple aggregation functions
        aggregated_df = df.groupby('p_np')['num'].agg(['count', 'mean'])

        print(grouped_df)
        print(aggregated_df)
    </code>
</pre>

---

**Pivot Tables and Reshaping Data**

**Explanation:**

Pivot tables help reorganize data to make it easier to analyze by converting rows into columns or vice versa. This is useful for summarizing large datasets into more meaningful information.

**Example Code:**

<pre>
    <code class="python">
        import pandas as pd

        # Example DataFrame
        data = {'Compound': ['A', 'B', 'A', 'B'],
                'Property': ['MeltingPoint', 'MeltingPoint', 'BoilingPoint', 'BoilingPoint'],
                'Value': [120, 150, 300, 350]}
        df = pd.DataFrame(data)

        # Creating a pivot table
        pivot_df = df.pivot_table(values='Value', index='Compound', columns='Property')

        print(pivot_df)
    </code>
</pre>

**Practice Problem:**

1. Create a pivot table from the BBBP dataset to summarize the average `num` for each `p_np` group (permeable and non-permeable).
2. Use the `melt()` function to reshape the DataFrame, converting columns back into rows.

**Solution**
<pre>
    <code class="python">
        import pandas as pd

        # Loading the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Creating a pivot table for 'num' grouped by 'p_np'
        pivot_df = df.pivot_table(values='num', index='p_np', aggfunc='mean')

        # Reshaping the DataFrame using melt
        melted_df = df.melt(id_vars=['name'], value_vars=['num', 'p_np'])

        print(pivot_df)
        print(melted_df.head())
    </code>
</pre>

### 2.2.1.4 Working with NumPy Arrays

---

**Basic Operations and Mathematical Functions**

**Explanation:**

NumPy is a library for numerical computing in Python, allowing for efficient array operations, including mathematical functions like summing or averaging.

**Example Code:**

<pre>
    <code class="python">
        import numpy as np

        # Example array
        arr = np.array([1, 2, 3, 4, 5])

        # Basic operations
        arr_sum = np.sum(arr)
        arr_mean = np.mean(arr)

        print(f"Sum: {arr_sum}, Mean: {arr_mean}")
    </code>
</pre>

**Practice Problem:**

1. Create a NumPy array from the `num` column in the **BBBP** dataset.
2. Perform basic statistical operations like `sum`, `mean`, and `median` on the `num` array.

**Solution**
<pre>
    <code class="python">
        import pandas as pd
        import numpy as np

        # Load the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Create a NumPy array from the 'num' column
        num_array = np.array(df['num'])

        # Perform basic statistical operations
        num_sum = np.sum(num_array)
        num_mean = np.mean(num_array)
        num_median = np.median(num_array)

        print(f"Sum: {num_sum}, Mean: {num_mean}, Median: {num_median}")
    </code>
</pre>

---

**Indexing and Slicing**

**Explanation:**

NumPy arrays can be sliced to access subsets of data.

**Example Code:**

<pre>
    <code class="python">
        import numpy as np

        # Example array
        arr = np.array([10, 20, 30, 40, 50])

        # Slicing the array
        slice_arr = arr[1:4]

        print(slice_arr)
    </code>
</pre>

**Practice Problem:**

1. Create a NumPy array from the `num` column in the **BBBP** dataset.
2. Slice the array to extract every second element.
3. Reverse the array using slicing.

**Solution**
<pre>
    <code class="python">
        import pandas as pd
        import numpy as np

        # Load the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Create a NumPy array from the 'num' column
        num_array = np.array(df['num'])

        # Slice the array to extract every second element
        sliced_array = num_array[::2]

        # Reverse the array using slicing
        reversed_array = num_array[::-1]

        print(f"Sliced Array (every second element): {sliced_array}")
        print(f"Reversed Array: {reversed_array}")
    </code>
</pre>

---

**Reshaping and Broadcasting**

**Explanation:**

Reshaping changes the shape of an array, and broadcasting applies operations across arrays of different shapes.

**Example Code:**

<pre>
    <code class="python">
        import numpy as np

        # Example array
        arr = np.array([[1, 2, 3], [4, 5, 6]])

        # Reshaping the array
        reshaped_arr = arr.reshape(3, 2)

        # Broadcasting: adding a scalar to the array
        broadcast_arr = arr + 10

        print(reshaped_arr)
        print(broadcast_arr)
    </code>
</pre>

**Practice Problem:**

1. Reshape a NumPy array created from the `num` column of the **BBBP** dataset to a shape of `(5, 20)` (or similar based on the array length).
2. Use broadcasting to add 100 to all elements in the reshaped array.

**Solution**
<pre>
    <code class="python">
        import pandas as pd
        import numpy as np

        # Load the BBBP dataset
        df = pd.read_csv('BBBP.csv')

        # Create a NumPy array from the 'num' column
        num_array = np.array(df['num'])

        # Reshaping the array to a (5, 20) shape (or adjust based on dataset length)
        reshaped_array = num_array[:100].reshape(5, 20)

        # Broadcasting: adding 100 to all elements in the reshaped array
        broadcasted_array = reshaped_array + 100

        print("Reshaped Array:")
        print(reshaped_array)

        print("\nBroadcasted Array (after adding 100):")
        print(broadcasted_array)
    </code>
</pre>

### 2.2.2 Data Visualization

Data visualization is critical for interpreting data and uncovering insights. In this section, we’ll use Python’s visualization libraries to create various plots and charts.

#### 2.2.2.1 Introduction to Visualization Libraries

**Explanation:**
Python has several powerful libraries for data visualization, including **Matplotlib**, **Seaborn**, and **Plotly**.

- **Matplotlib**: A foundational library for static, animated, and interactive visualizations.
- **Seaborn**: Built on top of Matplotlib, Seaborn simplifies creating informative and attractive statistical graphics.
- **Plotly**: Allows for creating interactive, web-ready plots.

**Example Code:**
<pre>
    <code class="python">
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
    </code>
</pre>

---

#### 2.2.2.2 Line and Scatter Plots

**Explanation:**
Line and scatter plots are used to display relationships between variables. Line plots are commonly used for trend analysis, while scatter plots are useful for examining the correlation between two numerical variables.

**Example Code for Line Plot:**
<pre>
    <code class="python">
import matplotlib.pyplot as plt

# Example data
time = [1, 2, 3, 4, 5]
concentration = [0.5, 0.6, 0.7, 0.8, 0.9]

# Line plot
plt.plot(time, concentration, marker='o')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration over Time')
plt.show()
    </code>
</pre>

**Example Code for Scatter Plot:**
<pre>
    <code class="python">
import seaborn as sns
import pandas as pd

# Load sample dataset
df = pd.DataFrame({'MolecularWeight': [180, 200, 150, 170, 210],
                   'BoilingPoint': [100, 110, 95, 105, 120]})

# Scatter plot
sns.scatterplot(data=df, x='MolecularWeight', y='BoilingPoint')
plt.title('Molecular Weight vs Boiling Point')
plt.show()
    </code>
</pre>

---

#### 2.2.2.3 Histograms and Density Plots

**Explanation:**
Histograms display the distribution of a single variable by dividing it into bins, while density plots are smoothed versions of histograms that show the probability density.

**Example Code for Histogram:**
<pre>
    <code class="python">
import matplotlib.pyplot as plt

# Example data
data = [1.5, 2.3, 2.9, 3.2, 4.0, 4.5, 5.1, 5.5, 6.3, 6.8]

# Histogram
plt.hist(data, bins=5, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
    </code>
</pre>

**Example Code for Density Plot:**
<pre>
    <code class="python">
import seaborn as sns

# Density plot
sns.kdeplot(data, shade=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Density Plot')
plt.show()
    </code>
</pre>

---

#### 2.2.2.4 Box Plots and Violin Plots

**Explanation:**
Box plots show the distribution of data based on quartiles and are useful for spotting outliers. Violin plots combine box plots and density plots to provide more detail on the distribution’s shape.

**Example Code for Box Plot:**
<pre>
    <code class="python">
import seaborn as sns
import pandas as pd

# Sample data
df = pd.DataFrame({'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
                   'Value': [10, 15, 10, 20, 15, 25]})

# Box plot
sns.boxplot(data=df, x='Category', y='Value')
plt.title('Box Plot')
plt.show()
    </code>
</pre>

**Example Code for Violin Plot:**
<pre>
    <code class="python">
# Violin plot
sns.violinplot(data=df, x='Category', y='Value')
plt.title('Violin Plot')
plt.show()
    </code>
</pre>

---

#### 2.2.2.5 Heatmaps and Correlation Matrices

**Explanation:**
Heatmaps display data as a color-coded matrix. They are often used to show correlations between variables or visualize patterns within data.

**Example Code for Heatmap:**
<pre>
    <code class="python">
import seaborn as sns
import numpy as np
import pandas as pd

# Sample correlation data
data = np.random.rand(5, 5)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])

# Heatmap
sns.heatmap(df, annot=True, cmap='coolwarm')
plt.title('Heatmap')
plt.show()
    </code>
</pre>

**Example Code for Correlation Matrix:**
<pre>
    <code class="python">
# Correlation matrix of a DataFrame
corr_matrix = df.corr()

# Heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
    </code>
</pre>

---

#### 2.2.2.6 Interactive Plots with Plotly

**Explanation:**
Plotly provides a range of interactive charts that can be easily embedded in web applications. Interactive plots allow users to hover over data points and zoom in on sections of the plot.

**Example Code for Interactive Scatter Plot:**
<pre>
    <code class="python">
import plotly.express as px

# Sample data
df = pd.DataFrame({'MolecularWeight': [180, 200, 150, 170, 210],
                   'BoilingPoint': [100, 110, 95, 105, 120]})

# Interactive scatter plot
fig = px.scatter(df, x='MolecularWeight', y='BoilingPoint',
                 title='Molecular Weight vs Boiling Point')
fig.show()
    </code>
</pre>

### 2.2.3 Statistical Analysis Basics

Statistical analysis is essential for interpreting data and making informed conclusions. In this section, we’ll explore fundamental statistical techniques using Python, which are particularly useful in scientific research.

---

#### 2.2.3.1 Descriptive Statistics

**Explanation:**
Descriptive statistics summarize and describe the main features of a dataset. Common descriptive statistics include the mean, median, mode, variance, and standard deviation.

**Example Code:**
<pre>
    <code class="python">
import pandas as pd

# Load a sample dataset
df = pd.DataFrame({'MolecularWeight': [180, 200, 150, 170, 210],
                   'BoilingPoint': [100, 110, 95, 105, 120]})

# Calculate descriptive statistics
mean_mw = df['MolecularWeight'].mean()
median_bp = df['BoilingPoint'].median()
std_mw = df['MolecularWeight'].std()

print(f"Mean Molecular Weight: {mean_mw}")
print(f"Median Boiling Point: {median_bp}")
print(f"Standard Deviation of Molecular Weight: {std_mw}")
    </code>
</pre>

**Practice Problem:**
Calculate the mean, median, and variance for the `num` column in the BBBP dataset.

**Solution**
<pre>
    <code class="python">
import pandas as pd

# Load the BBBP dataset
df = pd.read_csv('BBBP.csv')

# Calculate mean, median, and variance
mean_num = df['num'].mean()
median_num = df['num'].median()
variance_num = df['num'].var()

print(f"Mean: {mean_num}, Median: {median_num}, Variance: {variance_num}")
    </code>
</pre>

---

#### 2.2.3.2 Probability Distributions

**Explanation:**
Probability distributions describe how values are distributed across a dataset. The normal distribution is a common distribution that is symmetric about the mean.

**Example Code for Normal Distribution:**
<pre>
    <code class="python">
import numpy as np
import matplotlib.pyplot as plt

# Generate data with a normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)

# Plot the histogram
plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Normal Distribution')
plt.show()
    </code>
</pre>

**Practice Problem:**
Generate a normally distributed dataset based on the mean and standard deviation of the `num` column in the BBBP dataset. Plot a histogram of the generated data.

**Solution**
<pre>
    <code class="python">
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the BBBP dataset
df = pd.read_csv('BBBP.csv')

# Generate normally distributed data based on 'num' column
mean_num = df['num'].mean()
std_num = df['num'].std()
normal_data = np.random.normal(mean_num, std_num, size=1000)

# Plot histogram
plt.hist(normal_data, bins=30, density=True, alpha=0.6, color='g')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distribution of num')
plt.show()
    </code>
</pre>

---

#### 2.2.3.3 Hypothesis Testing

**Explanation:**
Hypothesis testing is a method for testing a hypothesis about a population parameter. A t-test, for example, can help determine if the means of two groups are significantly different.

**Example Code for t-test:**
<pre>
    <code class="python">
from scipy.stats import ttest_ind
import pandas as pd

# Example data
group_a = [1.2, 2.3, 1.8, 2.5, 1.9]
group_b = [2.0, 2.1, 2.6, 2.8, 2.4]

# Perform t-test
t_stat, p_val = ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat}, P-value: {p_val}")
    </code>
</pre>

**Practice Problem:**
In the BBBP dataset, compare the mean `num` values between permeable (p_np=1) and non-permeable (p_np=0) compounds using a t-test.

**Solution**
<pre>
    <code class="python">
from scipy.stats import ttest_ind
import pandas as pd

# Load the BBBP dataset
df = pd.read_csv('BBBP.csv')

# Separate data by permeability
permeable = df[df['p_np'] == 1]['num']
non_permeable = df[df['p_np'] == 0]['num']

# Perform t-test
t_stat, p_val = ttest_ind(permeable, non_permeable)
print(f"T-statistic: {t_stat}, P-value: {p_val}")
    </code>
</pre>

---

#### 2.2.3.4 Correlation and Regression

**Explanation:**
Correlation measures the strength and direction of a relationship between two variables, while regression predicts the value of a dependent variable based on one or more independent variables.

**Example Code for Correlation and Linear Regression:**
<pre>
    <code class="python">
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Example data
df = pd.DataFrame({'MolecularWeight': [180, 200, 150, 170, 210],
                   'BoilingPoint': [100, 110, 95, 105, 120]})

# Calculate correlation
corr, _ = pearsonr(df['MolecularWeight'], df['BoilingPoint'])
print(f"Correlation: {corr}")

# Linear regression
X = df[['MolecularWeight']]
y = df['BoilingPoint']
model = LinearRegression().fit(X, y)
print(f"Regression coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
    </code>
</pre>

**Practice Problem:**
Calculate the correlation between `num` and `p_np` in the BBBP dataset. Then, perform a linear regression to predict `num` based on `p_np`.

**Solution**
<pre>
    <code class="python">
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Load the BBBP dataset
df = pd.read_csv('BBBP.csv')

# Calculate correlation
corr, _ = pearsonr(df['num'], df['p_np'])
print(f"Correlation between num and p_np: {corr}")

# Linear regression
X = df[['p_np']]
y = df['num']
model = LinearRegression().fit(X, y)
print(f"Regression coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
    </code>
</pre>

---

#### 2.2.3.5 ANOVA (Analysis of Variance)

**Explanation:**
ANOVA tests whether there are statistically significant differences between the means of three or more independent groups. It’s useful for analyzing the impact of categorical variables on continuous data.

**Example Code for ANOVA:**
<pre>
    <code class="python">
from scipy.stats import f_oneway

# Example data for three groups
group1 = [1.1, 2.2, 3.1, 2.5, 2.9]
group2 = [2.0, 2.5, 3.5, 2.8, 3.0]
group3 = [3.1, 3.5, 2.9, 3.6, 3.3]

# Perform ANOVA
f_stat, p_val = f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat}, P-value: {p_val}")
    </code>
</pre>

**Practice Problem:**
Group the `num` column in the BBBP dataset by the first digit of `num` (e.g., 1XX, 2XX, 3XX) and perform an ANOVA test to see if the mean values differ significantly among these groups.

**Solution**
<pre>
    <code class="python">
from scipy.stats import f_oneway
import pandas as pd

# Load the BBBP dataset
df = pd.read_csv('BBBP.csv')

# Group 'num' by the first digit
group1 = df[df['num'].between(100, 199)]['num']
group2 = df[df['num'].between(200, 299)]['num']
group3 = df[df['num'].between(300, 399)]['num']

# Perform ANOVA
f_stat, p_val = f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat}, P-value: {p_val}")
    </code>
</pre>

## 2.3 Representation

### 2.3.1 Smiles and

### 2.3.2 Smarts

### 2.3.3 Fingerprint

### 2.3.4 3D coordinate

### 2.3.5 [Rdkit](https://www.rdkit.org/docs/GettingStartedInPython.html)

## 2.4 Calculation on Representation

### 2.4.1 Frigerprint

### 2.4.2
