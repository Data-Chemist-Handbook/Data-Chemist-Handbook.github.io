---
title: 10. LLMs for Chemistry
author: Quang Dao
date: 2024-08-20
category: Jekyll
layout: post
---

Dataset: [ChemLLMBench – Tasks Overview](https://github.com/ChemFoundationModels/ChemLLMBench#-tasks-overview)

**Brief Introduction**

   Large Language Models, or LLMs, are advanced machine learning models capable of understanding and generating human-like text and answering QA questions based on extensive training data from diverse sources.

   In chemistry, these models assist researchers and students by automating information retrieval, predicting chemical reactions, and supporting molecular design tasks. Benchmarks such as ChemLLMBench provide datasets to evaluate the accuracy and reliability of LLMs in chemical contexts.

   So what exactly are LLMs and why are they so powerful? We will dive into this in this section.

## 10.1 Introduction of LLMs

   **What are LLMs?**

   - Large Language Models (LLMs) are smart computer systems like ChatGPT and Gemini. They learn from huge amounts of text data and can understand and generate human-like writing.

   **What can they do?**

   - These AI models can understand the meaning behind words, answer questions, and even come up with new ideas.
   - In chemistry, LLMs are helpful because they can:
      - Simplify complicated chemical information.
      - Make difficult tasks easier to handle.
      - Quickly find and organize important data.
      - Help automate tasks that might otherwise take scientists a long time to complete.
      - Summarize extensive research articles or reports succinctly.

   In short, LLMs help make complex information clearer and easier to work with, even if you don't have a chemistry background.

   → Such applications can substantially speed up research processes, improve accuracy, and open new avenues for innovative scientific discoveries.

## 10.2 Prompt Engineering

   **Definition:**  
   Prompt engineering involves carefully writing instructions (prompts) to guide LLMs to give accurate and useful answers. In simpler terms, it's how you ask questions or provide information to an AI system clearly and specifically.

   **Why is it important in chemistry?**  
   Chemists often ask AI models to help with complex tasks, such as predicting chemical reactions or interpreting experimental data. Clear and detailed prompts help the AI understand exactly what information you're looking for.

   **Format of a Good Prompt:**

   1. **General Instruction:** Briefly describes the task context or role the model is expected to perform.  
      _Example:_  
      > “You are an expert chemist. Given the **[Input Representation]**: **[Input Data]**, predict the **[Target Output]** using your domain knowledge. Provide **[Input Explanation]**, **[Output Explanation & Restrictions]**, and **[Few-shot examples]**.”

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
   
   **Tips:**  
   - **Clarity:** Be specific.  
   - **Context:** Include reaction conditions or setup.  
   - **Structure:** State expected response format.  
   - **Examples:** Use few-shot prompts when helpful.

## 10.3 Usage of LLM APIs

   Chemists can utilize APIs from OpenAI, Google, and others to:

   - Embed intelligent text-based queries in chemical software or LIMS.  
   - Automate literature reviews: find publications, extract data, summarize research.  
   - Build dashboards for interactive chemical data analysis with AI insights.

   **Setting Up OpenAI API (Google Colab)**

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
      ```python
      def similarity_score(expected, answer):
         return int(expected.lower() == answer.lower())
      ```

## 10.4 Interactive Programming

   Interactive programming means writing code incrementally with immediate feedback, typically in Jupyter Notebooks or Colab.

   **Colab Setup**

   1. **Add secret:**  
      - Sidebar → Secrets → Add new:  
      - Name: `OPENAI_API_KEY`  
      - Value: your key  
      - Enable notebook access  

   2. **Install libs:**  
      ```bash
      !pip install -q openai pandas
      ```  

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

   
   Example notebook: [Example notebook](https://colab.research.google.com/drive/1B8qFtN_mkEzX3BGaznvXIstn0P1w6yRP?usp=sharingw)
## 10.5 Quiz  

### **Factual Questions**

1. **Prompt engineering** is best described as:  
   - A. Picking the fastest GPU for running an LLM  
   - B. Carefully crafting instructions so an LLM returns the desired type of answer  
   - C. Compressing model weights to fit on your laptop  
   - D. Translating SMILES strings into plain‑text English sentences  

<details><summary>Answer</summary>

**B**

</details>

<details><summary>Explanation</summary>

Prompt engineering focuses on how you *ask* the question so the model’s output is accurate, relevant, and formatted correctly.

</details>

---

2. Which option *best* explains why **chain‑of‑thought prompting** can improve an LLM’s performance on chemistry word problems?  
   - A. It increases the model’s context‑window size  
   - B. It forces the model to reveal its hidden layers  
   - C. It encourages the model to generate intermediate reasoning steps rather than jumping to a final answer  
   - D. It automatically fine‑tunes the model on PubChem abstracts during inference  

<details><summary>Answer</summary>

**C**

</details>

<details><summary>Explanation</summary>

By requesting step‑by‑step reasoning, you reduce “answer‑in‑one‑shot” errors and let the model use its latent reasoning abilities.

</details>


---

3. Which line will **safely** retrieve your OpenAI key that you stored in Colab *Secrets*?  
   - A. `openai.api_key = os.getenv("OPENAI_API_KEY")`  
   - B. `openai.api_key = "OPENAI_API_KEY"`  
   - C. `openai.api_key = open("OPENAI_API_KEY").read()`  
   - D. `os.environ.clear()`  

<details><summary>Answer</summary>

**A**

</details>

<details><summary>Explanation</summary>

Using `os.getenv` avoids hard‑coding or printing the secret.

</details>

---

4. Few‑shot prompting most closely resembles which traditional ML technique?  
   - A. Bag‑of‑words vectorization  
   - B. **k‑Nearest Neighbours** with *k = the number of examples you show*  
   - C. Stochastic gradient descent  
   - D. Monte‑Carlo tree search  

<details><summary>Answer</summary>

**B**

</details>

<details><summary>Explanation</summary>

You’re showing labelled exemplars and asking the model to act like “the nearest example”.

</details>

---

### **Comprehension / Application Questions**

5. You need an LLM to predict the *major* product of nitrating toluene. Which prompt is **most likely** to give a correct, parsable result?  

   - A. `Nitrate toluene.`  
   - B.  
     ```  
     You are an expert chemist. Given the reactant SMILES "Cc1ccccc1" and the reagent "HNO3/H2SO4", output only the major product in SMILES notation. Do **not** add commentary.  
     ```  
   - C. Please talk about electrophilic aromatic substitution, its history, and finally somewhere mention the product.  
   - D. Translate this to Klingon, then predict the product: toluene + nitric acid.  

<details><summary>Answer</summary>

**B**

</details>

<details><summary>Explanation</summary>

It supplies role, input format, reagent, and output constraints—maximising parseability.

</details>

---

6. The API snippet below keeps returning *“Rate limit reached”*. What is the **first** practical fix?  
```python
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=1,
    max_tokens=4000,  # ← current
    top_p=1
)
```  
   - A. Drop `max_tokens` to something realistic (e.g. 1024)
   - B. Switch model to `"text-davinci-002"`  
   - C. Delete your billing info  
   - D. Set `temperature` to 0  

<details><summary>Answer</summary>

**A**

</details>

<details><summary>Explanation</summary>

Huge `max_tokens` values are throttled; lowering it reduces quota usage per call.

</details>

---

7. Writing an interactive Colab helper like `gpt_query()` is valuable mainly because it lets you:  
    - A. Circumvent token‑billing limits  
    - B. Iterate quickly with different prompts and see Markdown immediately
    - C. Compile Python into CUDA kernels  
    - D. Guarantee the model never hallucinates  

<details><summary>Answer</summary>

**B**

</details>

<details><summary>Explanation</summary>

The helper hides boilerplate and renders responses as nicely formatted markdown for rapid experimentation.

</details>


