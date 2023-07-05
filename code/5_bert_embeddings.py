# -*- coding: utf-8 -*-

#Import necessary packages
import re
import torch
import pickle
import numpy as np
import pandas as pd
from stop_words import get_stop_words
from rank_bm25 import BM25Okapi
import transformers
from transformers import BertTokenizer, BertModel, BertForTokenClassification
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import torch
from torch.utils.data import DataLoader, TensorDataset

# Custom functions
from bert_text_pre_processing import add_labels
from CRF_utils import sent2features

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}")

"""
Data processing
We will use the labeled and unlabeled data in order to assign a cluster to each of the tokens. Start with pre-processing.
"""

# Download all data
unlabeled_reviews_train = pd.read_csv('./Data/original/drugsComTrain_raw.csv')
unlabeled_reviews_test = pd.read_csv('./Data/original/drugsComTest_raw.csv')

print("yo")
labeled_drug_reviews = pd.read_csv("./Data/combined/combined_df_1.csv")
print("yo")
# Concatenate unlabeled reviews
unlabeled_drug_reviews = pd.concat([unlabeled_reviews_train, unlabeled_reviews_test], axis = 0)
unlabeled_drug_reviews.reset_index(drop=True, inplace=True)

unlabeled_drug_reviews.head()

labeled_drug_reviews.head()

combined_dataset = pd.concat([unlabeled_drug_reviews["review"], labeled_drug_reviews["text"]], axis = 0)#.to_frame()
combined_dataset.reset_index(drop=True, inplace=True)

# Convert into list
combined_dataset_list = combined_dataset.to_list()
combined_dataset_list = [str(elem) for elem in combined_dataset_list] # Some reviews are not strings for some reason.

"""
The pre-trained BERT model requires the data to be tokenized in the same way as the training data used for the model. Special considerations: 

1. Each sentence must start with "[CLS]" and end with "[SEP]". 
2. All sentences must have the same number of tokens: some will be padded, other truncated. 
3. Need attention mask to keep track of the true tokens and the padding ones. 
"""

# Initialize tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenises data and puts it in BERT-compatible form 
tokenized_sentences = tokenizer(combined_dataset_list, add_special_tokens = True, 
                         max_length = 506, padding='max_length', return_attention_mask = True,
                         return_tensors='pt', truncation=True)

# Create Dataset and DataLoader with our data
input_ids = tokenized_sentences["input_ids"]
token_type_ids = tokenized_sentences["token_type_ids"]
attention_mask = tokenized_sentences["attention_mask"]

dataset = torch.utils.data.TensorDataset(input_ids, token_type_ids, attention_mask)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 100, shuffle = False)

"""
Extracting the embeddings.

In BERT, each unique token will have a different embedding depending on the context in which it is presented. This leads to context-dependent segmentation, which proved to be an advantage, for example, for polysemic words. 
In this model, BERT embeddings will be further split into clusters, and those cluster assignments built on a larger corpus of data will serve as a richer feature representation of the text that needs to be classified.
Due to computational resources, we could not keep track of all the different embeddings for each unique token in our text corpus. We found that the tokens with the biggest number of distinct embeddings were stop words. Additionally, the mean Euclidean distance between different embeddings of each ADR token was found to be significantly smaller than to different words. 
So we set the embedding of each unique token to be the mean of the different embeddings encountered. 
"""

# Initialize BERT model 
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model = model.to(device)
model.eval()

# Find number of unique tokens
unique_tokens = torch.unique(tokenized_sentences["input_ids"])

# Initialize tensor where the mean embeddings for each unique token will be stored
unique_embeddings = torch.zeros((unique_tokens.shape[0], 768))
unique_tokens_counter = torch.zeros((unique_tokens.shape[0], 1))

# Keep track fo the index of each token in the unique_embeddings tensor
unique_tokens_dict = dict(zip(unique_tokens.numpy(), range(len(unique_tokens))))

with torch.no_grad():

  for i, (input_ids, token_type_ids, attention_mask) in enumerate(dataloader):

    input_ids, token_type_ids, attention_mask = input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)

    # Forward pass through BERT and store hidden layers
    outputs = model(input_ids, token_type_ids, attention_mask)
    hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0) # Stack hidden layers - (layers, n sentences, n tokens, vector dim)
    embeddings_flat = token_embeddings.reshape(13, -1, 768).permute(1,0,2) # Get rid of the sentence dimension. - (layers, total n tokens, vector dim)

    # Obtain the embeddings
    token_vecs_sum = torch.zeros((embeddings_flat.shape[0], embeddings_flat.shape[2]))

    # For each token, sum the representation of the last 4-layers of BERT
    for j, token in enumerate(embeddings_flat):

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum[j, :] = sum_vec

    # Update the embeddings for each unique token
    batch_unique_tokens = torch.unique(input_ids.to("cpu").flatten()) # Find the unique tokens in each batch 
    batch_pos_unique_tokens = [torch.where(input_ids.to("cpu").flatten() == i)[0] for i in batch_unique_tokens] # Find the positions in which each unique token appears in the batch

    pos_unique_tokens = [unique_tokens_dict[id.item()] for id in batch_unique_tokens] # Find the position of the batch unique tokens in the original unique_token array
    
    # Sum the distinct embeddings for each token 
    for pos, batch_pos in zip(pos_unique_tokens, batch_pos_unique_tokens):
      unique_embeddings[pos, :] += token_vecs_sum[batch_pos, :].sum(axis=0)
      unique_tokens_counter[pos, 0] += len(batch_pos)

# Take the mean of the embeddings by dividing by the number of times each token appeared in the sentences
unique_embeddings = unique_embeddings / unique_tokens_counter

# Find the tokens corresponding to the unique token ids in the corpus
BERT_vocab = list(tokenizer.vocab.keys())
unique_token_words = [BERT_vocab[i] for i in unique_tokens]
unique_token_words = np.array(unique_token_words)

# Save the unique tokens and corresponding embeddings
#np.savetxt("/content/drive/MyDrive/NLP Project/Data/BERT_embeddings/unique_embeddings", unique_embeddings)
#np.savetxt("/content/drive/MyDrive/NLP Project/Data/BERT_embeddings/unique_tokens", list(unique_tokens))
