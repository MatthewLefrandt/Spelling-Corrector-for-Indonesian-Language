# Contextual Spelling Corrector for Indonesian Text Preprocessing

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📌 Overview
This project presents a **Contextual Spelling Corrector for Indonesian Text**, leveraging **Transformer-based models (T5 and BART)** to enhance NLP preprocessing by correcting spelling errors. It aims to improve text quality for downstream NLP tasks such as **machine translation, sentiment analysis, and information retrieval**.

The study compares different Transformer architectures, evaluates their performance on the **SPECIL dataset**, and fine-tunes the **best-performing model** for optimal results.

## 🚀 Features
✅ Preprocessing pipeline for Indonesian text spelling correction  
✅ Implementation of **T5-base** and **BART-base** models  
✅ Fine-tuning Transformer models on the **SPECIL dataset**  
✅ Evaluation using **BLEU Score, Edit Distance, Exact Match, and Character Accuracy**  
✅ Trained model available on **Hugging Face** for easy deployment  

The trained spelling correction model developed in this research is accessible through [Hugging Face Model](https://huggingface.co/MatthewLefrandt/T5-for-Indonesian-Spelling-Corrector).

## 📊 Performance Summary
| Model       | BLEU  | Edit Distance | Exact Match | Character Accuracy |
|------------|------|--------------|------------|------------------|
| **T5-base**  | **0.9380** | **0.2533** | **0.8656** | **0.9638** |
| BART-base   | 0.9145 | 0.5853 | 0.7620 | 0.9302 |

## 📝 Dataset
We use the **SPECIL (Spell Error Corpus for Indonesian Language)** dataset.  
📌 **Source:** [SPECIL Dataset on Kaggle](https://www.kaggle.com/datasets/yanfiyanfi/specil-spell-error-corpus-for-indonesian-language)  
📌 The dataset contains **21,500 sentences** covering **six types of spelling errors**.

## 🎯 Usage
### Dependencies
This project requires the following libraries:
- `transformers==4.48.3`
- `torch==2.5.1+cu124`
- `pandas==2.2.2`
- `numpy`

### Running Inference
```python
from typo_corrector import SentenceProcessor
import pandas as pd

# Initialize processor
processor = SentenceProcessor()

# Example dataframe
data = {
    'No': [1, 2],
    'Sentences': ['saya makn nasi. sya lapar.', 'hotel yg itu kotor. tdak nyaman.']
}
df = pd.DataFrame(data)

# Apply spelling correction
df['Corrected'] = df['Sentences'].apply(processor.process_sentences)

print(df)
```

## 🤝 Authors
- **Matthew Lefrandt**
- **Elvina Benedicta Santoso**
- **Alexander Agung Santoso Gunawan**
- **Jeffrey Junior Tedjasulaksana**

## 📬 Contact
For inquiries, reach out via email: **matthew.lefrandt@binus.ac.id**  
Or connect on GitHub: [MatthewLefrandt](https://github.com/MatthewLefrandt)
 
---
🚀 **Transform Indonesian NLP with Advanced Spelling Correction!**  
⭐ **Star this repo if you find it useful!**  
