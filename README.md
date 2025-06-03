# 📰 DOJ News Article Cleaning with Stopword Removal (NLP Preprocessing)

This notebook demonstrates how to clean real-world text data using Natural Language Processing (NLP) techniques.  
We use a dataset of U.S. Department of Justice (DOJ) news articles (`combined.json`) and show how to remove stop words using **spaCy**.

Perfect for **beginners in NLP** who want to understand the basics of text preprocessing before moving to more advanced tasks like classification or topic modeling.

---

## 📂 Dataset Source

The dataset is a JSON file with one news article per line (`lines=True` in pandas).  
Each article includes:
- `id`: identifier or press release code
- `title`: headline of the article
- `contents`: full press release text
- `date`: publication date
- `topics`: tags or themes (may be empty)
- `components`: DOJ divisions involved (e.g., Environment, National Security)

---

## ❓ Why Preprocessing Matters

### 🧩 Problem:
Text data is often noisy and filled with common, less informative words like:
- *“the”*, *“is”*, *“was”*, *“and”*, *“just”*, etc.

These are called **stop words**, and they:
- Don’t carry much meaning
- Increase vector size in NLP models
- Add noise to downstream tasks like classification

### ✅ Solution:
We use **spaCy** to:
- Tokenize text
- Remove stop words and punctuation
- Return clean, useful text for modeling

---

## 🔧 Preprocessing Code

We first load the spaCy English language model:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
