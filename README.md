# B141R-Data-Mining-

# Corporate Data Mining Pipeline: Customer Intelligence Extraction via SteamSense

This repository contains the complete production-ready data mining implementation for project **SteamSense**, designed to transform massive volumes of unstructured customer feedback into actionable business intelligence. By executing a modular Knowledge Discovery in Databases (KDD) pipeline on **9.6M+ user reviews**, the system isolates churn risks, benchmarks advanced predictive classifiers, and maps hidden consumer complaints to directly guide product optimization and maximize enterprise sales.

## 📊 Core Business Value & Objectives
* **Sales Maximization:** Identifying core retention drivers and conversion factors directly from player experiences.
* **Product Optimization:** Automatically extracting technical software defects, game balancing issues, and optimization bottlenecks.
* **Operational Scale:** Utilizing chunk-based processing architectures to ingest and filter multi-gigabyte datasets without memory overhead.

## ⚙️ Data Mining Pipeline Architecture

The implementation follows a strict 5-stage sequential lifecycle:
1. **Data Ingestion & Filtering:** Parsing the 8GB raw CSV dataset using chunk-loops (`chunksize=100000`), filtering for targeted language profiles (`english`), and drawing a reproducible stratified random sample of 100,000 entries.
2. **Feature Engineering & NLP Preprocessing:** Cleaning text via structured regular expression (Regex) filters, lowercasing, word tokenization, sentiment-preserving stopword engineering, and morphological reduction using the NLTK WordNet Lemmatizer.
3. **Data Transformation & Vectorization:** Mapping categorical classes into structural binary target values (`0` for Negative, `1` for Positive) and converting normalized text tokens into a 5,000-dimensional TF-IDF feature space (including unigrams and bigrams).
4. **Predictive Modeling Benchmarking:** Testing and evaluating models across rule-based lexicons (VADER), classical statistical classifiers (Multinomial Naive Bayes, Logistic Regression), and deep contextual transformers (Fine-tuned BERT).
5. **Unsupervised Knowledge Extraction:** Deploying Latent Dirichlet Allocation (LDA) modeling via the `gensim` library to automatically group positive feedback and negative product complaints into actionable themes.
<img width="2085" height="881" alt="image" src="https://github.com/user-attachments/assets/3abcf075-e7e9-45ee-853a-a8005ba1c1b8" />


## 📈 Quantitative Performance & Model Benchmarks

The models were evaluated using a stratified 80/20 train/test split. The quantitative metrics recorded are as follows:

| Machine Learning Framework | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| VADER (Lexicon Baseline) | 0.7000 | 0.9800 | 0.7100 | 0.8206 |
| Multinomial Naive Bayes (TF-IDF) | 0.8931 | 0.9800 | 0.9100 | 0.9416 |
| Logistic Regression (TF-IDF) | 0.9012 | 0.9800 | 0.9100 | 0.9452 |
| **Fine-tuned BERT (Transformer)** | **0.9600** | **0.9800** | **1.0000** | **0.9900** |

### Key Technological Highlights:
* **The Baseline (VADER):** Showed decent zero-training speed but struggled heavily with false positives due to complex sentence structures and gaming slang.
* **Statistical Classifiers:** Logistic Regression demonstrated phenomenal cost-to-performance efficiency on the TF-IDF matrix, yielding an accuracy of 90.12%.
* **The Production Choice (BERT):** Fine-tuned on a dedicated T4 GPU session, the `bert-base-uncased` transformer achieved near-perfect classification stability (F1 = 0.99), proving its capability to robustly handle nuanced feedback, negation, and sarcasm.

## 🛠️ Tech Stack & Environment
* **Platform:** Google Colab (T4 GPU Accelerated Session Architecture)
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `transformers`, `torch`, `gensim`, `matplotlib`, `seaborn`

## 📂 Primary Source Attribution
* **Dataset:** Steam Reviews 2021 Dataset (9,600,543 human-annotated rows)
* **Resource URL:** https://www.kaggle.com/datasets/najzeko/steam-reviews-2021
