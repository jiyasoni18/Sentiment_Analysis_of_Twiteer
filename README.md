# Twitter Sentiment Analysis using ML & BERT  
A complete end-to-end sentiment analysis pipeline built using Twitter data, combining **traditional Machine Learning models (RF/SVM/Logistic Regression)** with a **fine-tuned BERT model** for high-accuracy predictions.

## Project Overview  
This project analyzes tweets and classifies them into **Positive**, **Neutral**, or **Negative** sentiment.  
Two approaches were implemented:

### **1️ Traditional ML Pipeline (TF-IDF + ML Models)**
- TF-IDF Vectorizer  
- Random Forest  
- Linear SVM  
- Logistic Regression  
- Train–Test Split applied correctly  
- All trained models were saved as .pkl files:
### Machine learning Models Trained:
- `optimized_logistic_regression.pkl`
- `random_forest_model.pkl`
- `linear_svm_model.pkl`
- `tfidf_vectorizer.pkl`
  
### **2️ Deep Learning Pipeline (BERT)**
- Pretrained BERT model 
- Tokenization using `AutoTokenizer`  
- Cleaned text → Tokenizer → BERT → Prediction  
- BERT outperformed classical models on difficult examples

# BERT handles:
- Long sentences  
- Sarcasm  
- Emotional expressions  
- Subtle negative/positive cues  

## Dataset  
The dataset comes from **Kaggle** and contains:  
- `text` (tweets)  
- `clean_text` (after preprocessing)  
- `sentiment` labels  
- label → numerical label (0 = Negative, 1 = Neutral, 2 = Positive)
There was no original label column — all labeling was done manually during preprocessing.

# Preprocessing Steps  
✔ URL removal  
✔ Punctuation & number removal  
✔ Lowercasing  
✔ Tokenization  
✔ Stopword removal  
✔ Cleaned text stored as new column: `clean_text`  


### **What They Do:**
Tweets → TF-IDF → ML Model → Sentiment Prediction

### BERT Pipeline  
Tweet → Tokenizer → BERT Embeddings → Classification Head → Sentiment


# How to Run the Project

Install dependencies: pip install numpy pandas scikit-learn nltk transformers torch

Place all model .pkl files in the project directory.

Run the notebook: jupyter notebook tweet_data.ipynb

Use: print_one_line_predictions([...])
to compare models.


