# Import packages
import re
import pickle
import scipy.stats
import numpy as np
import pandas as pd
import joblib 
from itertools import chain
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec
from stop_words import get_stop_words
import nltk
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

nltk.download('omw-1.4')
nltk.download('wordnet')

# Import custom functions
from bert_text_pre_processing import add_labels
from CRF_utils import sent2features, preprocessing

save_model = False

"""
Data Processing
The unsupervised learning part was completed above, with the unsupervised data from kaggle (https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) and the combined supervised data sets from different sources. 
"""

# Download all data
unlabeled_reviews_train = pd.read_csv('./Data/original/drugsComTrain_raw.csv')
unlabeled_reviews_test = pd.read_csv('./Data/original/drugsComTest_raw.csv')

labeled_drug_reviews = pd.read_csv("./Data/combined/combined_df_1.csv")

# Concatenate unlabeled reviews
unlabeled_drug_reviews = pd.concat([unlabeled_reviews_train, unlabeled_reviews_test], axis = 0)
unlabeled_drug_reviews.reset_index(drop=True, inplace=True)

# Create lists of reviews for both datasets
unlabeled_reviews_list = unlabeled_drug_reviews.review.to_list() # A lists of lists. Contains characters
labeled_reviews_list = labeled_drug_reviews.text.to_list()

labeled_reviews_list = [x for x in labeled_reviews_list if str(x) != 'nan'] # Get rid of nans

# Combine lists
review_list = unlabeled_reviews_list
review_list.extend(labeled_reviews_list)

# Tokenize reviews
preprocessed_reviews = [preprocessing(i) for i in review_list]

print(f"Reviews have been pre-processed")

"""
Embeddings and clustering**

1. Obtain embeddings using Word2Vec for the whole set of rewiews.
2. Obtain clusters using K-means with 150 clusters, same number as in the paper. 

After, save the models.
"""

# Create Word embedding and clusters. Takes 3 minutes
model = Word2Vec(sentences = preprocessed_reviews, vector_size= 150, min_count=1)

print(f"Embedding has been created")

# Obtain the vector representations of the words. It's a dictionary
word_vectors = model.wv

vocab = np.array(list(model.wv.key_to_index.keys()))
word_vecs = []

for word in vocab:
    word_vecs.append(word_vectors[word])
    
word_array = np.array(word_vecs)

kmeans = KMeans(n_clusters=150).fit(word_array)

print(f"Clusters retrieved.")

if save_model == True:
  model.save("./Models/word2vec.model")
  joblib.dump(kmeans, "./Models/model.pkl")

"""
Data processing for CRF

For each token in each review, create dictionary containing:
1. The three previous and following tokens. 
2. The respective clusters of the aforementioned tokens. 
"""

df_1 = pd.read_csv(r'./Data/combined/combined_df_1.csv')
df_2 = pd.read_csv(r'./Data/combined/combined_df_2.csv')

pre_processed = add_labels(df_1, df_2, 'other', 'text', 'txt_id', 'symptom', False)

# split dataset into training and test/val
np.random.seed(100)

train_df, not_train_df = train_test_split(pre_processed, test_size=0.2)
valid_df, test_df = train_test_split(not_train_df, test_size=0.5)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Create a list of sentences for each DataFrame. Remove sentences with less than 
# 6 tokens and make sure the labels are strings. 
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

# Get CRF features for the three sets
X_train = [sent2features(s, vocab,  kmeans.labels_) for s in train_sentences]
X_val = [sent2features(s, vocab,  kmeans.labels_) for s in val_sentences]
X_test = [sent2features(s, vocab,  kmeans.labels_) for s in test_sentences]


# Train CRF
crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train_CRF)

print(f"CRF trained.")

y_pred = crf.predict(X_test)

acc = metrics.flat_accuracy_score(y_test_CRF, y_pred)
recall = metrics.flat_recall_score(y_test_CRF, y_pred, pos_label='1')
precision = metrics.flat_precision_score(y_test_CRF, y_pred, pos_label='1')
F1 = metrics.flat_f1_score(y_test_CRF, y_pred, pos_label='1')

print(f"Accuracy - {acc}, Recall - {recall}, Precision - {precision}, F1 - {F1}.")

if save_model == True:
  joblib.dump(crf, "./Models/CRF_word2vec")
