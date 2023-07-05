# -*- coding: utf-8 -*-

#Import necessary packages
import re
import pickle
import numpy as np
import pandas as pd
import joblib
from stop_words import get_stop_words
from rank_bm25 import BM25Okapi
import torch
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
import transformers
from transformers import BertTokenizer, BertModel, BertForTokenClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# Custom functions
from bert_text_pre_processing import add_labels 
from CRF_utils import sent2features

"""The BERT embeddings and the corresponding clusters are extracted in a different notebook. In this one, just load results"""

# Load embeddings 
unique_embeddings = np.loadtxt("./Data/BERT_embeddings/unique_embeddings")
unique_tokens = np.loadtxt("./Data/BERT_embeddings/unique_tokens")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Initialize tokenizer
BERT_vocab = list(tokenizer.vocab.keys())
unique_token_words = [BERT_vocab[int(i)] for i in unique_tokens]
unique_token_words = np.array(unique_token_words)

print("Embeddings loaded.")

#Cluster BERT embeddings
k_means_BERT = KMeans(n_clusters=800, n_init="auto").fit(unique_embeddings)

print("Clusters found")

# Pre-processing data for the CRF
# load data
df_1 = pd.read_csv(r'./Data/combined/combined_df_1.csv')
df_2 = pd.read_csv(r'./Data/combined/combined_df_2.csv')
pre_processed = add_labels(df_1, df_2, 'BERT', 'text', 'txt_id', 'symptom', False)

# split dataset into training, validation and test
np.random.seed(100)
train_df, val_test_df = train_test_split(pre_processed, test_size=0.2)
valid_df, test_df = train_test_split(val_test_df, test_size=0.5)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Keep only sentences with more than 6 tokens and make sure all labels are strings. VERY inefficient
train_sentences, y_train_CRF = [], []
val_sentences, y_val_CRF = [], []
test_sentences, y_test_CRF = [], []

for sent, lab in zip(train_df.tokenized.to_list(), train_df.pre_processed_tokens.to_list()):
  if len(sent) >= 6 and len(lab) == len(sent):
    train_sentences.append(sent)
    y_train_CRF.append(np.array(lab, dtype = 'str').tolist())

for sent, lab in zip(valid_df.tokenized.to_list(), valid_df.pre_processed_tokens.to_list()):
  if len(sent) >= 6 and len(lab) == len(sent):
    val_sentences.append(sent)
    y_val_CRF.append(np.array(lab, dtype = 'str').tolist())

for sent, lab in zip(test_df.tokenized.to_list(), test_df.pre_processed_tokens.to_list()):
  if len(sent) >= 6 and len(lab) == len(sent):
    test_sentences.append(sent)
    y_test_CRF.append(np.array(lab, dtype = 'str').tolist())

X_train = [sent2features(s, unique_token_words,  k_means_BERT.labels_) for s in train_sentences]
X_val = [sent2features(s, unique_token_words,  k_means_BERT.labels_) for s in val_sentences]
X_test = [sent2features(s, unique_token_words,  k_means_BERT.labels_) for s in test_sentences]

# Train CRF
crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train_CRF)

# Print results
y_pred = crf.predict(X_test)

acc = metrics.flat_accuracy_score(y_test_CRF, y_pred)
recall = metrics.flat_recall_score(y_test_CRF, y_pred, pos_label='1')
precision = metrics.flat_precision_score(y_test_CRF, y_pred, pos_label='1')
F1 = metrics.flat_f1_score(y_test_CRF, y_pred, pos_label='1')

print(f"Accuracy - {acc}, Recall - {recall}, Precision - {precision}, F1 - {F1}.")

