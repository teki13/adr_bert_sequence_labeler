import pandas as pd
import nltk
import re
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')


# get ADR lexicon dataframe
adr_lexicon = pd.read_csv('Data/ADR_lexicon.txt', sep='\t', names=['id', 'reaction', 'source'])
lexicon_list = adr_lexicon.reaction.to_list()

# ger reviews daatframe
reviews_df = pd.read_csv('Data/combined/combined_df_1.csv')
reviews_list = reviews_df.text.to_list()
reviews_id = reviews_df.txt_id.to_list()

# get list of nan reviews
d = dict(zip(reviews_id,reviews_list))
list_nan = [key for key, value in d.items() if isinstance(value, float)]

# remove nan reviews from review dataframe
reviews_df = reviews_df[~reviews_df['txt_id'].isin(list_nan)]
reviews_list = reviews_df.text.to_list()
reviews_id = reviews_df.txt_id.to_list()

# remove nan reviews from adr dataframe
adr_df = pd.read_csv('Data/combined/combined_df_2.csv')
adr_df = adr_df[~adr_df['txt_id'].isin(list_nan)]

# build initial ADR dictionary
ADRs = {}
for i in reviews_id:
    ADRs[i] = adr_df.loc[adr_df['txt_id'] == i]['symptom'].to_list()

# get review IDs that are in review dataframe but not in adr dataframe
no_adr_reviews = [k for k, v in ADRs.items() if v in (None, "", [])]

# remove them
reviews_df = reviews_df[~reviews_df['txt_id'].isin(no_adr_reviews)]
reviews_list = reviews_df.text.to_list()
reviews_id = reviews_df.txt_id.to_list()

# 2058 after removing the not annotated ones

adr_df = adr_df[~adr_df['txt_id'].isin(no_adr_reviews)]

# build final ADR dictionary
ADRs = {}
for i in reviews_id:
    ADRs[i] = adr_df.loc[adr_df['txt_id'] == i]['symptom'].to_list()


def preprocessing(content, remove_sw):
    # convert the text to lowercase
    content = content.lower()

    # remove non-alphabetical characters
    regex = re.compile('[^a-z\s]+')
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


def get_scores(model, lexicon_list=lexicon_list, reviews_list=reviews_list,  threshold_bm25=0.78, threshold_tf_idf=0.32):

    preprocessed_ADRs = [preprocessing(i, remove_sw=True) for i in lexicon_list]
    preprocessed_reviews = [preprocessing(i, remove_sw=True) for i in reviews_list]
    total_no_adrs = len(lexicon_list)

    # choose model
    if model == 'bm25':
        bm25 = BM25Okapi(preprocessed_ADRs)

        precision_list = []
        recall_list = []
        f1_score_list = []
        accuracy_list = []
        no_retrieved_adrs = []

        for i, j in zip(range(len(preprocessed_reviews)), reviews_id):

            # get actual ADRs
            actual_ADRs = ADRs[j]

            #get the scores for every ADRs for this specific review
            score_list = bm25.get_scores(preprocessed_reviews[i])

            # build dataframe with scores
            scores_df = pd.DataFrame({'ADR': lexicon_list, 'score': score_list})

            # remove rows with score=0
            scores_df = scores_df[scores_df['score'] != 0]

            # normalize scores
            scores_df['normalized_score'] = (scores_df['score'] - scores_df.score.min()) / (scores_df.score.max() - scores_df.score.min())

            # keep only scores over thershold
            scores_df = scores_df[scores_df['normalized_score'] > threshold_bm25]
            no_retrieved_adrs.append(len(scores_df))
            
            # get final list of ADRs obtained from model
            model_ADRs = scores_df.ADR.to_list()

            # compute metrics
            TP = 0
            FP = 0
            for i in model_ADRs:
                if i in actual_ADRs: TP += 1
                else: FP += 1

            FN = len(actual_ADRs) - TP

            no_actual = [i for i in lexicon_list if i not in actual_ADRs]
            TN_list = [i for i in no_actual if i not in model_ADRs]
            TN = len(TN_list)

            recall = TP / (TP + FN)

            if TP + FP != 0: precision = TP / (TP + FP)
            else: precision = 0

            if TP != 0: f1_score = 2 * ((precision * recall) / (precision + recall))
            else: f1_score = 0

            accuracy = (TP + TN) / total_no_adrs

            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            accuracy_list.append(accuracy)
        
        # buid results dataframe
        results_df = pd.DataFrame({'review_id': reviews_id, 'precision': precision_list, 'recall': recall_list, 
                                  'f1_score': f1_score_list, 'accuracy': accuracy_list})
        
        precision = sum(results_df.precision.to_list()) / len(results_df.precision.to_list())
        recall = sum(results_df.recall.to_list()) / len(results_df.recall.to_list())
        f1 = sum(results_df.f1_score.to_list()) / len(results_df.f1_score.to_list())
        accuracy = sum(results_df.accuracy.to_list()) / len(results_df.accuracy.to_list())
        print('Number of ADRs retrieved for each review', no_retrieved_adrs)
        print('Average number of ADRs retrieved:', sum(no_retrieved_adrs)/len(no_retrieved_adrs))

        # return results
        return 'BM25 results: ', 'precision:', precision, 'recall:', recall, 'f1-score:', f1, 'accuracy', accuracy
    
    elif model == 'tf_idf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(lexicon_list)

        precision_list = []
        recall_list = []
        f1_score_list = []
        accuracy_list = []
        no_retrieved_adrs = []

        for i, j in zip(range(len(reviews_list)), reviews_id):

            # get actual ADRs
            actual_ADRs = ADRs[j]

            query_tfidf = vectorizer.transform([reviews_list[i]])

            # get scores
            cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

            # build dataframe with scores
            scores_df = pd.DataFrame({'ADR': lexicon_list, 'score': cosine_similarities})

            # keep only scores over thershold
            scores_df = scores_df[scores_df['score'] > threshold_tf_idf]
            no_retrieved_adrs.append(len(scores_df))
            
            # get final list of ADRs obtained from model
            model_ADRs = scores_df.ADR.to_list()

            # compute metrics
            TP = 0
            FP = 0
            for i in model_ADRs:
                if i in actual_ADRs: TP += 1
                else: FP += 1

            FN = len(actual_ADRs) - TP

            no_actual = [i for i in lexicon_list if i not in actual_ADRs]
            TN_list = [i for i in no_actual if i not in model_ADRs]
            TN = len(TN_list)

            recall = TP / (TP + FN)

            if TP + FP != 0: precision = TP / (TP + FP)
            else: precision = 0

            if TP != 0: f1_score = 2 * ((precision * recall) / (precision + recall))
            else: f1_score = 0

            accuracy = (TP + TN) / total_no_adrs

            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            accuracy_list.append(accuracy)

        # buid results dataframe
        results_df = pd.DataFrame({'review_id': reviews_id, 'precision': precision_list, 'recall': recall_list, 
                                  'f1_score': f1_score_list, 'accuracy': accuracy_list})
        
        precision = sum(results_df.precision.to_list()) / len(results_df.precision.to_list())
        recall = sum(results_df.recall.to_list()) / len(results_df.recall.to_list())
        f1 = sum(results_df.f1_score.to_list()) / len(results_df.f1_score.to_list())
        accuracy = sum(results_df.accuracy.to_list()) / len(results_df.accuracy.to_list())
        #print('Number of ADRs retrieved for each review', no_retrieved_adrs)
        print('Average number of ADRs retrieved:', sum(no_retrieved_adrs)/len(no_retrieved_adrs))

        # return results
        return 'TF-IDF results: ', 'precision:', precision, 'recall:', recall, 'f1-score:', f1, 'accuracy', accuracy


def get_ADRs_for_new_review(model, review, lexicon_list=lexicon_list, threshold_bm25=0.78, threshold_tf_idf=0.32):

    preprocessed_ADRs = [preprocessing(i, remove_sw=True) for i in lexicon_list]
    tokenized_review = preprocessing(review, remove_sw=True)

    # choose model
    if model == 'bm25':
        bm25 = BM25Okapi(preprocessed_ADRs)

        # get the scores for every ADRs for this specific review
        score_list = bm25.get_scores(tokenized_review)

        # build dataframe with scores
        scores_df = pd.DataFrame({'ADR': lexicon_list, 'score': score_list})

        # remove rows with score=0
        scores_df = scores_df[scores_df['score'] != 0]

        # normalize scores
        scores_df['normalized_score'] = (scores_df['score'] - scores_df.score.min()) / (scores_df.score.max() - scores_df.score.min())

        # keep only scores over thershold
        scores_df = scores_df[scores_df['normalized_score'] > threshold_bm25]
            
        # get final list of ADRs obtained from model
        model_ADRs = scores_df.ADR.to_list()
    
    elif model == 'tf_idf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(lexicon_list)

        query_tfidf = vectorizer.transform([review])

        # get scores
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        # build dataframe with scores
        scores_df = pd.DataFrame({'ADR': lexicon_list, 'score': cosine_similarities})

        # keep only scores over thershold
        scores_df = scores_df[scores_df['score'] > threshold_tf_idf]
            
        # get final list of ADRs obtained from model
        model_ADRs = scores_df.ADR.to_list()

    return model_ADRs

print(get_scores('tf_idf'))
print(get_scores('bm25'))

# print(get_ADRs_for_new_review('tf_idf','i was suffering from insomnia before starting taking some medicine. I did not feel any dizziness in the morning as reported in the leaflet, but sometimes i have been feeling nauseous when i wake up.'))
# print(get_ADRs_for_new_review('bm25', 'i was suffering from insomnia before starting taking some medicine. I did not feel any dizziness in the morning as reported in the leaflet, but sometimes i have been feeling nauseous when i wake up.'))









