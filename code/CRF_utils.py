# Now implement CRF
# Now need to create the features to feed the CRF
import numpy as np 
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocessing(content, remove_sw = False):

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

def find_cluster(word, vocab, cluster_ids, max_cluster = 150):
    word_ind = np.where(word == vocab)[0]
    if len(word_ind)!= 0:
      return cluster_ids[word_ind][0]
    else:
      return max_cluster


# Obtain the CRF features given a sentence from the training set
def word2features(sent, i, vocab, cluster_ids, max_cluster = 150):

    word = sent[i]
    # Start by initializing position independent features
    features = {"word": word, 
                "wordC": find_cluster(word, vocab, cluster_ids, max_cluster)}
    L = len(sent)
    
    if (i>2) and (i < (L-3)):

        word1, word2, word3 = sent[i+1], sent[i+2], sent[i+3]
        wordm3, wordm2, wordm1 = sent[i-3], sent[i-2], sent[i-1]
        
        features.update({'wordm3': wordm3, 'wordm2': wordm2 ,'wordm1': wordm1,
                        'word1': word1,'word2': word2,'word3': word3,
                        'wordm3C': find_cluster(wordm3, vocab, cluster_ids, max_cluster), 
                        'wordm2C': find_cluster(wordm2, vocab, cluster_ids, max_cluster), 
                        'wordm1C': find_cluster(wordm1, vocab, cluster_ids, max_cluster), 
                        'word3C': find_cluster(word3, vocab, cluster_ids, max_cluster), 
                        'word2C': find_cluster(word2, vocab, cluster_ids, max_cluster), 
                        'word1C': find_cluster(word1, vocab, cluster_ids, max_cluster)})
    elif i < 3: 
        # Store full future
        word1, word2, word3 = sent[i+1], sent[i+2], sent[i+3]
        features.update({'word1': word1,'word2': word2,'word3': word3,
                         'word3C': find_cluster(word3, vocab, cluster_ids, max_cluster), 
                         'word2C': find_cluster(word2, vocab, cluster_ids, max_cluster), 
                         'word1C': find_cluster(word1, vocab, cluster_ids, max_cluster)})
        if i > 0:
            wordm1 = sent[i-1]
            features.update({'wordm1': wordm1,
                         'wordm1C': find_cluster(wordm1, vocab, cluster_ids, max_cluster)})
            if i==2:
                wordm2 = sent[i-2]
                features.update({'wordm2':wordm2, 
                                 'wordm2C': find_cluster(wordm2, vocab, cluster_ids, max_cluster)})
        else: 
            features.update({'BOS': True})
    
    elif i > (L-4):
        # Store full past
        wordm1, wordm2, wordm3 = sent[i-1], sent[i-2], sent[i-3]
        features.update({'wordm1': wordm1,'wordm2': wordm2,'wordm3': wordm3,
                         'wordm3C': find_cluster(wordm3, vocab, cluster_ids, max_cluster), 
                         'wordm2C': find_cluster(wordm2, vocab, cluster_ids, max_cluster), 
                         'wordm1C': find_cluster(wordm1, vocab, cluster_ids, max_cluster)})
        if i < (L-1):
            word1 = sent[i+1]
            features.update({'word1': word1,
                             'word1C': find_cluster(word1, vocab, cluster_ids, max_cluster)})
            if i == (L - 3):
                word2 = sent[i+2]
                features.update({'word2': word2,
                                 'word2C': find_cluster(word2, vocab, cluster_ids, max_cluster)})
        else: 
            features.update({'EOS': True})
            
    return features

# Get all the features of each word in a sentence in a list
def sent2features(sent, vocab, cluster_ids, max_cluster = 150):
  return [word2features(sent, word_id, vocab, cluster_ids, max_cluster) for word_id in range(len(sent))]
