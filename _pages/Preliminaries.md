---
title: 2. Preliminaries
author: Haomin, Dan Yoo
date: 2024-08-12
category: Jekyll
layout: post
---

In the Preliminaries character, we will introduce some base coding still and data analysis still for the beginners.

## 2.1 Introduction to Python

Here is some python coding content.

### 2.1.1 Setting up Python environment

Anaconda, Jupyter Notebook

### 2.1.2 Basic syntax and structure

variables, loops, functions

### 2.1.3 Python packages

josn

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

You have been provided with a CSV file named `molecule_data.csv` containing molecular weights and melting points of various compounds. Write Python code to:

1. Read the CSV file into a DataFrame.
2. Display the first five rows of the DataFrame.
3. Calculate the average molecular weight of the compounds.

**Answer:**

<pre>
    <code class="python">
import pandas as pd

# 1. Read the CSV file
df = pd.read_csv('molecule_data.csv')

# 2. Display the first five rows
print(df.head())

# 3. Calculate the average molecular weight
average_mw = df['MolecularWeight'].mean()
print(f"Average Molecular Weight: {average_mw}")
    </code>
</pre>

---

#### Handling Chemical Data Formats Using RDKit

**Explanation:**

Chemical data often come in specialized formats like SDF (Structure Data File), MOL, or SMILES strings. RDKit is a powerful cheminformatics library in Python that allows you to read and manipulate these chemical file formats.

**Example Code:**

<pre>
    <code class="python">
from rdkit import Chem
from rdkit.Chem import PandasTools

# Reading an SDF file into a list of molecule objects
supplier = Chem.SDMolSupplier('compounds.sdf')
molecules = [mol for mol in supplier if mol is not None]

# Converting SDF to a Pandas DataFrame
df = PandasTools.LoadSDF('compounds.sdf', smilesName='SMILES', molColName='Molecule')

# Display the DataFrame
print(df.head())
    </code>
</pre>

**Explanation of the Code:**

- `Chem.SDMolSupplier()` reads molecules from an SDF file.
- `PandasTools.LoadSDF()` loads the SDF file into a DataFrame, adding SMILES representations and molecule objects.

**Practice Problem:**

You have an SDF file named `drug_candidates.sdf` containing several potential drug molecules. Write Python code to:

1. Read the SDF file into a DataFrame.
2. Add a new column that calculates the LogP (a measure of hydrophobicity) for each molecule.
3. Display the first five entries of the DataFrame, showing the molecule name and calculated LogP.

**Answer:**

<pre>
    <code class="python">
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors

# 1. Read the SDF file into a DataFrame
df = PandasTools.LoadSDF('drug_candidates.sdf', smilesName='SMILES', molColName='Molecule')

# 2. Calculate LogP for each molecule
df['LogP'] = df['Molecule'].apply(Descriptors.MolLogP)

# 3. Display the first five entries
print(df[['MoleculeName', 'LogP']].head())
    </code>
</pre>

---

#### Importing Data from Databases

**Explanation:**

Data is often stored in databases, especially in large organizations or collaborative projects. Python can connect to databases using libraries like `sqlite3` for SQLite databases or `SQLAlchemy` for more complex scenarios. For chemical data, databases like PubChem can be accessed via their APIs.

**Example Code:**

<pre>
    <code class="python">
import sqlite3
import pandas as pd

# Connecting to a SQLite database
conn = sqlite3.connect('chemistry.db')

# Reading data from a SQL table into a DataFrame
df = pd.read_sql_query("SELECT * FROM compounds", conn)

# Display the DataFrame
print(df.head())

# Don't forget to close the connection
conn.close()
    </code>
</pre>

**Explanation of the Code:**

- `sqlite3.connect()` establishes a connection to the database.
- `pd.read_sql_query()` executes an SQL query and reads the result into a DataFrame.
- Always close the database connection when done.

**Practice Problem:**

Suppose you have a database `lab_results.db` with a table `assays` containing bioassay results for various compounds. Write Python code to:

1. Connect to the database.
2. Retrieve all records where the activity is greater than 50.
3. Load the data into a DataFrame and display the number of records retrieved.

**Answer:**

<pre>
    <code class="python">
import sqlite3
import pandas as pd

# 1. Connect to the database
conn = sqlite3.connect('lab_results.db')

# 2. Retrieve records with activity > 50
query = "SELECT * FROM assays WHERE activity > 50"
df = pd.read_sql_query(query, conn)

# 3. Display the number of records
print(f"Number of records retrieved: {len(df)}")

# Close the connection
conn.close()
    </code>
</pre>


### 2.2.2 Data visualization


### 2.2.3 Statistical analysis basics

## 2.3 Representation

### 2.3.1 Smiles and

### 2.3.2 Smarts

### 2.3.3 Fingerprint

### 2.3.4 3D coordinate

### 2.3.5 [Rdkit](https://www.rdkit.org/docs/GettingStartedInPython.html)

## 2.4 Calculation on Representation

### 2.4.1 Frigerprint

### 2.4.2
