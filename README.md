# 📰 Fake News Detection in Arabic and English

A Natural Language Processing (NLP) project to classify news articles as **fake** or **real** in **both Arabic and English** using advanced transformer-based models. This system helps combat misinformation by analyzing news content through fine-tuned models like CNN and GRU, offering robust multilingual support.

---


## 📌 Overview
This project detects fake news using:
- Deep Learning models
- TF-IDF with classical ML algorithms
- Data preprocessing pipelines for both English and Arabic
- Evaluation metrics like accuracy, precision, recall, and F1-score

---

## ⚙️ Tech Stack
- Python
- scikit-learn
- pandas, NumPy
- Google Colab / Jupyter Notebook

---

## ✨ Features
- 🔤 Multi-language support (Arabic and English)
- 🧹 Arabic preprocessing (normalization, lemmatization, stopword removal)
- 🤖 Model comparison: classical ML vs. transformer-based models
- 📊 Confusion matrix and metric-based evaluation

---
## 🧠 Models Used

### Classical Machine Learning Models:
- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes
- Random Forest Classifier
- XGBoost Classifier

### Deep Learning Models:
- CNN (using `Conv1D`)
- LSTM
- GRU
- BiLSTM (Bidirectional LSTM)
- BiGRU (Bidirectional GRU)

---

## 🧠 Methodology

The project follows a two-branch methodology for fake news detection in both **Arabic** and **English**:

### 🔄 1. Data Preprocessing
- **Arabic:**
  - Text normalization (removing diacritics, elongation, punctuation)
  - Stopword removal
  - Tokenization
- **English:**
  - Lowercasing and punctuation removal
  - Tokenization TF-IDF vectorization
  - Stopword filtering using NLTK

### 2. Feature Representation

- Classical models: **TF-IDF Vectorization**
- Deep Learning models: **Word Embeddings** using `Embedding` layers

### 🧪 3. Model Training 
- Classical models trained with `scikit-learn` & `xgboost`
- Deep learning models built using `Keras` with architectures like:
  - CNN (Conv1D + GlobalPooling)
  - LSTM, GRU, and their Bidirectional versions
- Trained with categorical cross-entropy loss and early stopping


### 📊 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📈 Results: 
Unigram for ML

| Model Type   | Architecture             | Language | Accuracy | F1 Score |
|--------------|---------------------------|----------|----------|----------|
| Classical    | Logistic Regression       | English  | 0.92     | 0.92     |
| Classical    | SVM(LinearSVC)            | English  | 0.93     | 0.93     |
| Classical    | Multinomial NB            | English  | 0.89     | 0.89     |
| Classical    | Random Forest             | English  | 0.91     | 0.91     |
| Classical    | XGBoost                   | English  | 0.92     | 0.93     |
| Deep Learning| BiLSTM                    | English   | 0.93    
| Deep Learning| BiGRU                     | English   | 0.92    
| Deep Learning| CNN                       | English   | 0.92    
| Deep Learning| GRU                       | English  | 0.92 
| Deep Learning| LSTM                      | English  | 0.86

> Among all models, **BiGRU** and **BiLSTM** performed the best for English and Arabic respectively.

---

## ⚠️ Limitations

- ⚖️ **Imbalanced datasets** may affect classification performance, particularly recall on the minority class.
- 💻 **Computational requirements** are high for transformer models, especially on large datasets.
- 🌐 **Arabic NLP** remains complex due to rich morphology and lack of high-quality pre-cleaned datasets.
- 🔍 Model interpretability is limited — transformers are often black boxes.

---

## 🔮 Future Work

- 🧠 Integrate **Explainable AI (XAI)** techniques (e.g., LIME, SHAP) to enhance transparency in predictions.
- 🌍 Extend support to other languages (e.g., French, German) with multilingual transformers like XLM-R.
- 🌐 Deploy the model using **Flask** or **Streamlit** for interactive fake news classification.
- 🧪 Experiment with **ensemble techniques** that combine classical and deep models.
- 📊 Incorporate **real-time news streams** for live detection and performance testing.


## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/Mayonaisey/fake-news-nlp.git
cd fake-news-nlp
