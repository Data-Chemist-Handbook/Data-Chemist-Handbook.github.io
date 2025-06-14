---
title: 10. LLMs for Chemistry
author: Quang Dao
date: 2024-08-20
category: Jekyll
layout: post
---
Large Language Models, or LLMs, are advanced machine learning models capable of understanding and generating human-like text and answering QA questions based on extensive training data from diverse sources.

In chemistry, these models assist researchers and students by automating information retrieval, predicting chemical reactions, and supporting molecular design tasks. Benchmarks such as ChemLLMBench provide datasets to evaluate the accuracy and reliability of LLMs in chemical contexts.

So what exactly are LLMs and why are they so powerful? We will dive into this in this section.

## 10.1 Introduction of LLMs

### What are LLMs?

- Large Language Models (LLMs) are smart computer systems like ChatGPT and Gemini. They learn from huge amounts of text data and can understand and generate human-like writing.

### What can they do?

- These AI models can understand the meaning behind words, answer questions, and even come up with new ideas.
- In chemistry, LLMs are helpful because they can:
- Simplify complicated chemical information.
- Make difficult tasks easier to handle.
- Quickly find and organize important data.
- Help automate tasks that might otherwise take scientists a long time to complete.
- Summarize extensive research articles or reports succinctly.

In short, LLMs help make complex information clearer and easier to work with, even if you don't have a chemistry background.

→ Such applications can substantially speed up research processes, improve accuracy, and open new avenues for innovative scientific discoveries.

---

### Section 10.1 - Quiz Questions
#### 1) Factual Questions

##### Question 1) 
LLM is an acronym for:  
**A.** Limited‑Length Memory  
**B.** Large Language Model  
**C.** Layered Lattice Matrix  
**D.** Linear‑Loss Machine    

<details>
<summary>▶ Click to show answer</summary>

Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

An LLM is a Large Language Model trained on vast text corpora to understand and generate human‑like language.
</details>

---

##### Question 2) 
Which capability is **not** typically associated with an LLM used in chemistry?  
**A.** Summarising a 50‑page synthetic protocol  
**B.** Translating SMILES to IUPAC  
**C.** Running *ab initio* molecular‑orbital calculations  
**D.** Answering literature questions in natural language  

<details>
<summary>▶ Click to show answer</summary>
Correct Answer: C
</details>

<details>
<summary>▶ Click to show explanation</summary>

LLMs work with text; high‑level quantum calculations require specialised physics engines, not language models.

</details>

---

#### 2) Comprehension / Application Question

##### Question 3) 
You ask an LLM: “List three advantages of using ChatGPT when preparing a lab report.”  
Which response below shows that the model has understood the chemistry context rather than giving generic writing advice?  
**A.** Use active voice, avoid jargon, add headings.   
**B.** Quickly draft the experimental *Procedure* section, check safety phrases (e.g., ‘corrosive,’ ‘flammable’), and suggest ACS‑style citations for each reagent.  
**C.** Provide colourful images, add emojis, write shorter paragraphs.   
**D.** Increase your font size to 14 pt.   

<details>
<summary>▶ Click to show answer</summary>
Correct Answer: B
</details>

<details>
<summary>▶ Click to show explanation</summary>

Only option B references elements specific to chemistry reports (procedure, hazards, ACS citations).

</details>



## 10.2 Prompt Engineering

### Definition:
Prompt engineering involves carefully writing instructions (prompts) to guide LLMs to give accurate and useful answers. In simpler terms, it's how you ask questions or provide information to an AI system clearly and specifically.

### Why is it important in chemistry?
Chemists often ask AI models to help with complex tasks, such as predicting chemical reactions or interpreting experimental data. Clear and detailed prompts help the AI understand exactly what information you're looking for.

### Format of a Good Prompt:

1. **General Instruction:** Briefly describes the task context or role the model is expected to perform.  
_Example:_  “You are an expert chemist. Given the **[Input Representation]**: **[Input Data]**, predict the **[Target Output]** using your domain knowledge. Provide **[Input Explanation]**, **[Output Explanation & Restrictions]**, and **[Few-shot examples]**.”

2. **Input Representation:** Type of input (e.g., SMILES, molecular description).  
_Example:_ “SMILES string”, “molecular structure”.

3. **Input Data:** The specific data instance the model will work with.  
_Example:_ `CC(=O)OC1=CC=CC=C1C(=O)O` (Aspirin) or “benzene + nitric acid”.

4. **Target Output:** Expected prediction/result.  
_Example:_ “product SMILES”.

5. **Relevant Domain:** The chemical expertise area to use.  
_Example:_ “organic chemistry”.

6. **Input Explanation:** Clarifies provided data.  
_Example:_ “Given the reactants aspirin and hydrochloric acid...”

7. **Output Explanation & Restrictions:** Desired response format and limits.  
_Example:_ “Provide the reaction products in SMILES notation only.”

8. **Few-shot Examples:** Demonstrate the expected format.  
```text
Input: Reactants: aspirin + hydrochloric acid  
Output (SMILES): <product_smiles>

Input: Reactants: benzene + nitric acid  
Output (SMILES): <product_smiles>
```

**Example Task and Prompt:**

**Task:** Analyze the interaction between hydrogen gas and water.  
> “You are an expert chemist. Explain what happens when hydrogen gas (H₂) is introduced to water (H₂O) under standard conditions. Include:
> -Physical and chemical properties of hydrogen in water  
> -Possible reactions or interactions  
> -Conditions for reaction  
> -Energy changes or hazards  
> -Explanation of why the reaction proceeds or not under normal conditions  
> Provide relevant safety considerations.  
> **Example:** Analyze the interaction between chlorine gas and water.  
> **Answer:**  
> Chlorine gas is a yellow‑green diatomic molecule with moderate solubility in water (≈0.7 g/L at 20 °C) and a strong oxidizing character. When Cl₂ dissolves, it undergoes a hydrolysis equilibrium:Cl₂(g) + H₂O(l) ⇌ HCl(aq) + HOCl(aq). Under ambient conditions this reaction is mildly exothermic, releasing enough heat to warm the solution but  not boil it. Hydrochloric acid and hypochlorous acid lower the pH, driving further Cl₂ uptake until equilibrium is reached. Hypochlorous acid is an effective disinfectant, owing to its oxidizing power. The reaction proceeds spontaneously because Cl₂ is a strong electrophile and water acts as a nucleophile, facilitating bond cleavage. Hazards include toxicity and corrosiveness of both gaseous Cl₂ and its acidic products; HOCl can bleach skin or materials. Always handle chlorine in a fume hood, wear acid‑resistant gloves, goggles, and respiratory protection, and neutralize any acidic effluent before disposal.”

### Tips: 
- **Clarity:** Be specific.  
- **Context:** Include reaction conditions or setup.  
- **Structure:** State expected response format.  
- **Examples:** Use few-shot prompts when helpful.

---

### Section 10.2 - Quiz Questions
#### 1) Factual Questions

##### Question 1) 
In a well‑structured prompt, the portion *“SMILES string: CCO”* is best labelled as:  
**A.** General Instruction  
**B.** Input Data  
**C.** Relevant Domain   
**D.** Few‑shot Example   

<details><summary>▶ Click to show answer</summary>

Correct Answer: B

</details>

<details><summary>▶ Click to show explanation</summary>

It supplies the concrete data instance the model must process.

</details>

---

##### Question 2) 
Few‑shot examples are included primarily to:  
**A.** Reduce token count  
**B.** Demonstrate the output format and reasoning style the model should mimic  
**C.** Hide the system instruction  
**D.** Increase randomness  

<details><summary>▶ Click to show answer</summary>

Correct Answer: B

</details>

<details><summary>▶ Click to show explanation</summary>

By pattern‑matching the examples, the model generalises to the new query.

</details>

---

#### 2) Comprehension / Application Questions

##### Question 3
You write the single‑line prompt: “Predict the major product.”  
Your LLM responds with a 700‑word essay and no product structure. Which prompt‑engineering fix is most direct?  
**A.** Add more temperature detail.   
**B.** Specify “Output only a SMILES string, no commentary.”   
**C.** Increase the temperature parameter to 2.   
**D.** Split the prompt into two requests.   

<details><summary>▶ Click to show answer</summary>

Correct Answer: B

</details>

<details><summary>▶ Click to show explanation</summary>

Explicit output restrictions steer the model to a parsable answer.

</details>



## 10.3 Usage of LLM APIs

Chemists can utilize APIs from OpenAI, Google, and others to:

- Embed intelligent text-based queries in chemical software or LIMS.  
- Automate literature reviews: find publications, extract data, summarize research.  
- Build dashboards for interactive chemical data analysis with AI insights.

### Setting Up OpenAI API (Google Colab)

1. **Open a notebook** at https://colab.research.google.com  
2. **Install packages:**  
```bash
!pip install openai 
!pip install pandas
!pip install numpy
!pip install kagglehub
```  
3. **Authenticate:**  
```python
import os, openai
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")
```  
4. **Load data:**  
```python
import kagglehub, pandas as pd
path = kagglehub.dataset_download("priyanagda/bbbp-smiles")
df = pd.read_csv(f"{path}/BBBP.csv")
```  
_Or, if not on Kaggle:_  
```python
import requests, os, pandas as pd
url = "URL_TO_FILE"
os.makedirs("data", exist_ok=True)
file_path = "data/file.csv"
with open(file_path, "wb") as f:
f.write(requests.get(url).content)
df = pd.read_csv(file_path)
```  
5. **Make an API call:**  
```python
from openai import OpenAI
client = OpenAI()
sample_data = df.to_csv(index=False)
prompt = f"YOUR QUESTION HERE. Reference data:\n{sample_data}"
completion = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{"role": "system", "content": "You are an expert chemist. Respond in Markdown."},
{"role": "user",   "content": prompt}
],
temperature=1,
max_tokens=1000,
top_p=1
)
```  
6. **Evaluate:**  
Call this function with two arguments, the model’s predicted answer and the ground-truth solution, to compute the model’s accuracy.
```python
def similarity_score(expected, answer):
return int(expected.lower() == answer.lower())
```
### Section 10.3 – Quiz Questions

#### 1) Factual Questions

##### Question 1  

In the Colab setup, the purpose of  
```python
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```  
is to:  
**A.**  Download a dataset from Kaggle   
**B.**  Make the key available to the `openai` library inside the session’s environment  
**C.**  Encrypt your code cell   
**D.**  Select the GPT‑4 model   

<details><summary>▶ Click to show answer</summary>

Correct Answer: B

</details>

<details><summary>▶ Click to show explanation</summary>

The key is stored as an environment variable so the SDK can read it without hard‑coding.

</details>

---

##### Question 2  
Which argument in `client.chat.completions.create()` *mainly* controls result length?  
**A.** `temperature`  
**B.** `top_p`  
**C.** `max_tokens` 
**D.** `model`  

<details><summary>▶ Click to show answer</summary>

Correct Answer: C

</details>

<details><summary>▶ Click to show explanation</summary>

`max_tokens` sets the upper bound on generated tokens, limiting output size.

</details>

---

#### 2) Comprehension / Application Questions

##### Question 3
You call the Chat Completion API twice with the **exact same prompt** but get **different** answers each time.  
Which settings will make the response **identical** on every call?

**A.** `temperature = 1.0`, `top_p = 1.0`  
**B.** `temperature = 0.0`, `top_p = 1.0`  
**C.** `temperature = 0.0`, `top_p = 0.0`
**D.** `temperature = 0.5`, `top_p = 0.5`  

<details><summary>▶ Click to show answer</summary>

**C**

</details>

<details><summary>▶ Click to show explanation</summary>
Setting both temperature and top_p to 0 removes all randomness, forcing the model to choose the highest‑probability token at each step—so identical prompts yield identical outputs.
</details>



## 10.4 Interactive Programming

Interactive programming means writing code incrementally with immediate feedback, typically in Jupyter Notebooks or Colab.

### Colab Setup

1. **Add secret:**  
- Sidebar → Secrets → Add new:  
- Name: `OPENAI_API_KEY`  
- Value: your key  
- Enable notebook access  

### Prepare Library 

2. **Install libs:**  
```bash
!pip install -q openai
!pip install -q pandas
```  

### Coding

3. **Helper function:**  
```python
import os
import pandas as pd
from google.colab import userdata
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
client = OpenAI()

def gpt_query(prompt, file_path=None):
data_info = ""
if file_path and file_path.endswith(".csv"):
df = pd.read_csv(file_path)
data_info = f"\nDataset sample:\n{df.head()}\n"
completion = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{"role": "system", "content": "You are a helpful assistant. Respond in Markdown."},
{"role": "user",   "content": prompt + data_info}
],
temperature=1,
max_tokens=1000,
top_p=1
)
return completion.choices[0].message.content.strip()
```

4. **Run query:**  
```python
from IPython.display import Markdown, display
output = gpt_query(prompt=input("Enter a prompt: "))
display(Markdown(output))
```  

### Section 10.4 – Quiz Questions

#### 1) Factual Questions

##### Question 1  
Storing the API key in Colab “Secrets” is safer than typing it in plain text because:  
**A.** It shortens execution time.  
**B.** The key is hidden in the notebook UI and not saved to revision history.
**C.** It gives you free credits.  
**D.** The model runs locally.  

<details><summary>▶ Click to show answer</summary>

Correct Answer: B

</details>

<details><summary>▶ Click to show explanation</summary>

Secrets are masked and excluded from shared notebook diffs, reducing accidental leakage.

</details>

---

#### 2) Comprehension / Application Questions

##### Question 2
The helper `gpt_query()` function optionally prints `df.head()` when given a `.csv` path.  
Why is showing the *head* helpful before the API call?  
**A.** To check column names and detect obvious parsing errors.  
**B.** To compute cosine similarity.  
**C.** To save network bandwidth.  
**D.** To anonymise the dataset.  

<details><summary>▶ Click to show answer</summary>

Correct Answer: A

</details>

<details><summary>▶ Click to show explanation</summary>

A quick visual snapshot lets you verify data integrity before you embed it in a prompt.

</details>

