#import libraries
import pandas as pd
import torch
import transformers
from transformers import BertTokenizer, BertModel, BertForTokenClassification
import numpy as np
import nltk
import re
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import nltk
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
nltk.download('wordnet')
nltk.download('omw-1.4')


#Functions

#Function that checks if an ADR is in text
def is_included(row):
    list1, list2 = row['tokenized'], row['tokenized_adr']
    result = []

    i = 0
    while i < len(list1):

        if list1[i:i+len(list2)] == list2:
          for j in range(len(list2)):
            result.append(1)
            i += 1
        else:
            result.append(0)
            i += 1
            
    return result


# define a function to sum the lists
def sum_lists(lst):
    return [sum(x) for x in zip(*lst)]



#convert to BERT format
def convert_BERT_format(df_1, column):

  # add the symbol [CLS] as the beginning of each sentence
  df_1[column] = df_1.apply(lambda x: '[CLS]' + ' ' + str(x[column]), axis = 1)
  # add the symbol [SEP] at the end of each sentence in the review 
  df_1[column] = df_1.apply(lambda x: str(x[column]).replace('.', '. [SEP]'), axis = 1)

  return df_1


def remove_whitespace(df_1, column):
  
  #remove \n and \t (for some reason we have to do this when we load back in the data)
  df_1[column] = df_1.apply(lambda x: str(x[column]).replace("\n", " "), axis = 1)
  df_1[column] = df_1.apply(lambda x: str(x[column]).replace("\t", " "), axis = 1)


def preprocessing(content, remove_sw):
    # convert the text to lowercase
    content = content.lower()
    regex = re.compile('[^a-z\s]+')

    # remove all commas so that constructions such as $70,000 maintain their meaning and do not get split:'70', '000'
    content = regex.sub('', content)

    # https://www.adamsmith.haus/python/answers/how-to-remove-all-punctuation-marks-with-nltk-in-python
    # remove punctuation and tokenize (which will be the same as 1-grams)
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    one_grams = tokenizer.tokenize(content)

    #remove stopwords
    if remove_sw == True:
        one_grams = [i for i in one_grams if i not in get_stop_words('english')]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    words = []
    for word in one_grams:
        words.append(lemmatizer.lemmatize(word))   

    return words


def add_labels(df_1, df_2, tokenizer_type, column, id_column, symptom, remove_stopw):
  '''
  A function that geiven input of a review and symptoms, returns a preprocessed data set
  where we have the review tokenized, and a corresponding columns that shows at which position
  an ADR can be dfound. For examle if we have "My myscles hurt", the returned value would be
  one column ['my', 'muslces', 'hurt'] and another column [0,1,0] with the 1 indicating the position
  of the ADR
  Inputs:
        df_1 - dataframe which contains the reviwes (dataframe)
        df_2 - a dataframe that contains the ADRs (dataframe)
        tokenizer_type - input "BERT" for bert tokenizer and "Other" for regular (str)
        column - the column in which the text review is found in df_1 dataset (str)
        id_column - the column by which df_1 and df_2 can be merged (str)
        symptom - the name of the column which contains the AD in df_2 (str)
        remove_stopw - set True to remove and False not to remove (boolen)
                      Note: for BERT, we never remove the stop words
  Output:
        preprocess_data - the datframe whith the processed reviews, and correposing
        location of the ADR (dataframe)
  '''
  #remove white space
  remove_whitespace(df_1, column)
  
  #BERT Tokenizer
  if tokenizer_type == "BERT":
    convert_BERT_format(df_1, column)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    df_1['tokenized'] = df_1.apply(lambda x: tokenizer.tokenize(x[column]), axis =1)
    df_2['tokenized_adr'] = df_2.apply(lambda x: tokenizer.tokenize(str(x[symptom])), axis =1)

  #Non-BERT tokenizer
  else:

    
    #convert the columns to list
    df_1_list = df_1[column].to_list()
    df_2_list = df_2[symptom].to_list()

    #preprocess the text
    preprocessed_1 = [preprocessing(str(i), remove_sw=remove_stopw) for i in df_1_list]
    preprocessed_2 = [preprocessing(str(i), remove_sw=remove_stopw) for i in df_2_list]

    df_1['tokenized'] = preprocessed_1
    df_2['tokenized_adr'] = preprocessed_2


  #merge the 2 dataframes
  merged_df = df_1.merge(df_2, on= id_column, how='left')
  merged_df = merged_df[merged_df['tokenized_adr'].notna()]

  merged_df['token_included'] = merged_df.apply(is_included, axis=1)

  # group the dataframe by "group_col" and apply the "sum_lists" function to "Tokens"
  result = merged_df.groupby(id_column)['token_included'].apply(sum_lists).reset_index()

  # rename the columns in the result dataframe
  result.columns = [id_column, 'pre_processed_tokens']

  #merge back with the rest of the data to get final output
  preprocess_data = df_1.merge(result, on=id_column, how='left')
  preprocess_data = preprocess_data[preprocess_data['pre_processed_tokens'].notna()]
  
  #replace all values higher than 1 with 0
  list_tokenized =  preprocess_data['pre_processed_tokens'].to_list()
  for i in list_tokenized:
    for j in range(len(i)):

      if i[j] > 1: 
        i[j] = 1

  return preprocess_data


def split_long_reviews(tokens, labels, txt_id_l, dataset_l, text_l):

  '''
  This functiont akes as input the subset of the dataframe to be split into 
  smaller subset if the maximum length of the input size is exceeded

  Inputs:
    tokens - a list of the column that includes the tokenized text (list)
    lables - a list of the column that includes the labels (list)
    txt_id_1 - a list of the column which includes the text id (list)
    dataset_1 - a list of the column that includes the dataset name (list)
    text_l - a list of the column that includes the text (list)
  
  Outoput:
    dubset_df - a dataframe which has split the long texts into smaller ones
                presented as a dataframe of the same format
  '''

  new_tokens = []
  new_labels = []
  new_txt_id = []
  new_dataset = []
  new_text = []

  for i in range(len(tokens)):

    #split them the tokens and the labels in half

    tokens_1 = tokens[i][:len(tokens[i])//2]
    tokens_2 = tokens[i][len(tokens[i])//2:]

    labels_1 = labels[i][:len(labels[i])//2]
    labels_2 = labels[i][len(labels[i])//2:]

    #add the [SEP] token to the tokens_1 and [CLS] to tokens_2
    tokens_1.append('[SEP]')
    tokens_2.insert(0, '[CLS]')

    labels_1.append(0)
    labels_2.insert(0, 0)

    #append the newly created tokens
    new_tokens.append(tokens_1)
    new_tokens.append(tokens_2)

    new_labels.append(labels_1)
    new_labels.append(labels_2)

    new_txt_id.append(txt_id_l[i] + '_1')
    new_txt_id.append(txt_id_l[i] + '_2')

    new_dataset.append(dataset_l[i])
    new_dataset.append(dataset_l[i])

    new_text.append(text_l[i])
    new_text.append(text_l[i])

  #create a dataframe with these new inputs
  subset = {'txt_id': new_txt_id, 'text': new_text, 'dataset': new_dataset, 'tokenized': new_tokens, 'pre_processed_tokens': new_labels}
  subset_df = pd.DataFrame(data=subset)

  #return the new dataframe with split columns
  return subset_df



def introduce_padding(max_len, t, symbol):
  '''
  A function that pads the inputs.

  Inputs: max_len - the maximum length until which to pad (int)
          t - the list to be added with padding (list)
          symbol - the symbol to be padded with. For example
          if we are padding a list of tokens this would be [PAD]
          if we are padding a list of labels this would be 0 (int/str)
  
  Output: t - the padded list (list)
  '''
  #get the current length of the list
  t_len = len(t)

  #find the difference between the max len an the list len
  diff = max_len - t_len

  #padd in the place of all the difference
  for i in range(diff):
    t.append(symbol)

  #return the padded list
  return t


def convert_data_to_tensor(df, label, mask, token_id):

  ''''
  This function converts the necessary data for train/test
  and converts it into tensors

  Inputs: df - the dataframe which contains the columns (dataframe)
          label - the label column (str)
          mask - the column which contains the mask (str)
          token_id - the column which contains the column id (str)
  Outputs:
          padded_att_mask - tensor of the mask matrix (tensor)
          padded_token_ids - tensor of the token matrix (tensor)
          padded_labels - tensor of the label matrix (tensor)
  '''

  #get the data needed for the data loader (test)
  padded_att_mask = np.stack(df[mask].values, axis=0)
  padded_token_ids = np.stack(df[token_id].values, axis=0)
  padded_labels = np.stack(df[label].values, axis=0)

  #convert the data to tensor (test)
  padded_att_mask = torch.from_numpy(padded_att_mask)
  padded_token_ids = torch.from_numpy(padded_token_ids)
  padded_labels = torch.from_numpy(padded_labels)

  return padded_att_mask, padded_token_ids, padded_labels



def evaluate_model(dataloader_test, model): 

  '''
  A function that evaluates the performance of the model
  It tests it on the test datasets and returns the 
  predicted and true labels

  Inputs: dataloader_test - dataloader which contains the test data
          model - the trained model 
  
  Outputs: predictions - an n x m numpy array which contains the predictions (numpy array)
           true_labels - an n x m numpy array with contains the true labels (numpy array)
  '''

  model.eval()
    
  eval_loss = 0
  predictions = np.array([], dtype = np.int64).reshape(0, max_len)
  true_labels = np.array([], dtype = np.int64).reshape(0, max_len)


  with torch.no_grad():
    for i, (padded_att_mask, padded_token_ids, padded_labels) in enumerate(dataloader_test):

      #set to available device
      padded_att_mask = padded_att_mask.to(device)
      padded_token_ids = padded_token_ids.to(device)
      padded_labels = padded_labels.to(device)

      #make predictions
      output = model(padded_token_ids, 
                        token_type_ids=None,
                        attention_mask=padded_att_mask,
                        labels=padded_labels)
      

      step_loss = output[0]
      eval_prediction = output[1]

      eval_loss += step_loss

      eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis = 2)
      actual = padded_labels.to('cpu').numpy()

      predictions = np.concatenate((predictions, eval_prediction), axis = 0)
      true_labels = np.concatenate((true_labels, actual), axis = 0)


    return predictions, true_labels



#load the combined files
df_1 = pd.read_csv(r'Data/combined/combined_df_1.csv')
df_2 = pd.read_csv(r'Data/combined/combined_df_2.csv')

#add the labels to the data
pre_processed = add_labels(df_1, df_2, 'BERT', 'text', 'txt_id', 'symptom', False)

#get the length of each of the reviews
pre_processed['r_lenght'] = pre_processed.apply(lambda x: len(x.tokenized), axis = 1)
#find the maximum lenthg of the tokens
max_len = max(pre_processed['r_lenght'])
print("Max length is", max_len)

#There are 7 reviews that have tokens larger than the max input size of BERT. 
reviews_long = pre_processed[pre_processed['r_lenght'] >= 512]

#split the tokens which have size of more than 512 in half
tokens = reviews_long['tokenized'].to_list()
labels = reviews_long['pre_processed_tokens'].to_list()
txt_id_l = reviews_long['txt_id'].to_list()
dataset_l = reviews_long['dataset'].to_list()
text_l = reviews_long['text'].to_list()

split_df = split_long_reviews(tokens, labels, txt_id_l, dataset_l, text_l)

#include a column for the length
split_df['r_lenght'] = split_df.apply(lambda x: len(x.tokenized), axis = 1)

# remove the longer queries from the final data
preprocessed_same_length = pre_processed[pre_processed['r_lenght'] < 512]
#append the new splitdataframe to this dataframe
preprocessed_comibined = pd.concat([preprocessed_same_length, split_df]).reset_index()

#add an attention mask such that all tokens are 1 and later when we padd the input we itnroduce 0 for the padded portions
preprocessed_comibined['att_mask'] = preprocessed_comibined.apply(lambda x: x.r_lenght * [1], axis = 1)

#Introduce padding to the inputs
max_len = max(preprocessed_comibined['r_lenght']) #the max length for the padding would be the longest list
#padd the tokens
preprocessed_comibined['padded_tokens'] = preprocessed_comibined.apply(lambda x: introduce_padding(max_len, x.tokenized, '[PAD]'), axis = 1)
#padd the labels
preprocessed_comibined['padded_labels'] = preprocessed_comibined.apply(lambda x: introduce_padding(max_len, x.pre_processed_tokens, 2), axis = 1)
#padd the attention mask
preprocessed_comibined['padded_att_mask'] = preprocessed_comibined.apply(lambda x: introduce_padding(max_len, x.att_mask, 0), axis = 1)

#double check the length to confirm that the padding was done correctly
preprocessed_comibined['new_len'] = preprocessed_comibined.apply(lambda x: len(x.padded_tokens), axis = 1)

#Introduce the token ids into the dataset

#define the tokenizer one more time
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#get the token ids for each of the tokens
preprocessed_comibined['token_ids'] = preprocessed_comibined.apply(lambda x: tokenizer.convert_tokens_to_ids(x.padded_tokens), axis = 1)

#Split the dataset into test and train
np.random.seed(100)
train_df, valid_df = train_test_split(preprocessed_comibined, test_size=0.3)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print(pre_processed.shape)
print(train_df.shape, valid_df.shape)

#get the train data as tensors
padded_att_mask_train, padded_token_ids_train, padded_labels_train = convert_data_to_tensor(train_df,'padded_labels', 'padded_att_mask', 'token_ids')
#get the test data as tensors
padded_att_mask_test, padded_token_ids_test, padded_labels_test = convert_data_to_tensor(valid_df,'padded_labels', 'padded_att_mask', 'token_ids')

#create data loaders

#train loader
dataset = TensorDataset(padded_att_mask_train, padded_token_ids_train, padded_labels_train)
dataloader_train = DataLoader(dataset, batch_size=10, shuffle=True)

#test loader
dataset = TensorDataset(padded_att_mask_test, padded_token_ids_test, padded_labels_test)
dataloader_test = DataLoader(dataset, batch_size=10, shuffle=False)

# Initialize the model
model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased',  num_labels = 3)

#check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#define the optimizer
params = model.parameters()
optimizer = torch.optim.Adam(params, lr= 3e-5)

#Train the model
epoch = 1

for i in range(epoch):

  train_loss = 0

  for j, (padded_att_mask, padded_token_ids, padded_labels) in enumerate(dataloader_train):

    #set to available device
    padded_att_mask = padded_att_mask.to(device)
    padded_token_ids = padded_token_ids.to(device)
    padded_labels = padded_labels.to(device)

    #make predictions
    output = model(padded_token_ids, 
                       token_type_ids=None,
                       attention_mask=padded_att_mask,
                       labels=padded_labels)
    

    step_loss = output[0]
    prediction = output[1]
        
    step_loss.sum().backward()
    optimizer.step()        
    train_loss += step_loss
    optimizer.zero_grad()

  print(f"Epoch {i} , Train loss: {train_loss.sum()}")


#save the model 
torch.save(model.state_dict(), 'Models/bert_model_4.pt')

#load model
model.load_state_dict(torch.load('/Models/bert_model_4.pt', map_location=torch.device('cpu')))

#evaluate model
predictions, true_labels = evaluate_model(dataloader_test, model)

#remove the paddings before calculating the evaulation metrics
true_labels_2classes = []
true_pred_2classes = []

for i in range(len(true_labels)):

  for j in range(len(true_labels[i])):

    if true_labels[i][j] != 2:
      true_labels_2classes.append(true_labels[i][j])
      true_pred_2classes.append(predictions[i][j])



true_final_labels = []
pred_final_labels = []

for i in range(len(true_pred_2classes)):

  if true_pred_2classes[i] != 2:
    true_final_labels.append(true_labels_2classes[i])
    pred_final_labels.append(true_pred_2classes[i])


#Calculate the f1 score, percision, recall and accuracy

#calculate the f1score
f1_score_r = f1_score(true_final_labels, pred_final_labels, pos_label = 1)

#calculate percision and recall
precision = precision_score(true_final_labels, pred_final_labels, pos_label = 1)
recall = recall_score(true_final_labels, pred_final_labels, pos_label = 1)

#calculate the accuracy
accuracy = accuracy_score(true_final_labels, pred_final_labels)

print("The F1 score is", f1_score_r)
print("Precision is", precision)
print("Recall is", recall)
print("Acuracy is", accuracy)







