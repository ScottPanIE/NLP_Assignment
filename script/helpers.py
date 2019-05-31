# Collection of helper functions for NLP

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from nltk import word_tokenize, WordPunctTokenizer, pos_tag
from nltk.corpus import stopwords
import nltk
SPLITS = 5

def rearrange(df):
    for row in range(len(df)):
        if not pd.isna(df.iloc[row, 5]):
            # Concatenate according columns
            df.iloc[row, 1] = df.iloc[row, 1] + df.iloc[row, 2] + df.iloc[row, 3]
            df.iloc[row, 2] = df.iloc[row, 4]
            df.iloc[row, 3] = df.iloc[row, 5]
        elif not pd.isna(df.iloc[row, 4]):
            df.iloc[row, 1] = df.iloc[row, 1] + df.iloc[row, 2]
            df.iloc[row, 2] = df.iloc[row, 3]
            df.iloc[row, 3] = df.iloc[row, 4]
    df = df.drop(['X1', 'X2'], axis = 1)
    return df

def append_scores(prev_scores, name, cv_scores, test_score):
    prev_scores = prev_scores.append({'Model': name, 'CV Score avg': np.mean(cv_scores), 'CV Score max': np.max(cv_scores), 'Holdout Score': test_score}, ignore_index = True)
    return prev_scores

def cv_evaluate(X, y, X_val, y_val, model, folds = SPLITS):
    cv_scores = cross_val_score(model,
                                X,
                                y,
                                cv = folds)
    model.fit(X, y)
    pred = model.predict(X_val)
    test_score = accuracy_score(y_val, pred)
    print(pd.crosstab(y_val, pred))
    return cv_scores, test_score

stopwords_en = stopwords.words('english')
# define function to eliminate the stopwords
def eliminate_stopwords(wordslist):
    """
    stopwords_en is predefined outside of the function
    """
    wordslist = [i for i in wordslist if i.isalpha()]
    clean_list = [i for i in wordslist if i not in stopwords_en]
    return clean_list

# postag
def count_postags(words):
    tagged_words = nltk.pos_tag(words)
    num_noun = 0
    num_verb = 0
    num_adj = 0
    for word in tagged_words:
        if word[1] == 'NN':
            num_noun += 1
        elif word[1] == 'VERB':
            num_verb += 1
        elif word[1] == 'ADJ':
            num_adj += 1
    return num_noun, num_verb, num_adj

# detect keywords
def detect_keyword(words, keyword):
    if keyword in words:
        return 1
    else:
        return 0

def count_date(x):
    datelist = re.findall(r'\w+\s\d{1,2},\s\d{4}', str(x))
    if len(datelist) > 0:
        dateinfo = " HAS_DATE"
    else:
        dateinfo = " NO_DATE"
    return x + dateinfo

def prepare_data(data, tokenize = False):
    """
    preparing data, fixing shape issues
    """
    # fix X1 X2 issue
    # 'if' needed since the submission dataset does not have the next two features
    if 'X1' in data.columns:
        data = rearrange(data)
    if 'label' in data.columns:
        data.label = data.label.apply(lambda x: 1 if x == 'REAL' else 0)
    
    # remove \n symbol and extract date information
    data['text_edit'] = data.text.apply(lambda x: re.sub("\\n", "", str(x)))
    data['text_edit'] = data.text_edit.apply(lambda x: count_date(x))

    if tokenize:
        data['title'] = data.title.apply(lambda x:" ".join(word_tokenize(x.lower())))
        data['text'] = data.text.apply(lambda x:" ".join(word_tokenize(x.lower())))
    
    return data


def preprocess_data(data, drop_unprocessed = True):
    """
    preprocessing + feature engineering
    """
    # tokenize the title and text
    data['title_token'] = data.title.apply(lambda x:word_tokenize(x.lower()))
    data['text_token'] = data.text.apply(lambda x:word_tokenize(x.lower()))
    
    # eliminate the stopwords in title and text
    data['titletoken_without_stopwords'] = data.title_token.apply(lambda x:eliminate_stopwords(x))
    data['texttoken_without_stopwords'] = data.text_token.apply(lambda x:eliminate_stopwords(x))
    # drop the redundent features
    if drop_unprocessed:
        data = data.drop(['text','title'],axis=1)
    # need to eliminate the punctuation as well? 
    # maybe do it with regex from the original title and text

    ## feature engineering 
    
    # find keywords
    data['trump_title'] = data.titletoken_without_stopwords.apply(lambda x:detect_keyword(x,'trump'))
    data['trump_text'] = data.texttoken_without_stopwords.apply(lambda x:detect_keyword(x,'trump'))
    
    # count the postags
    # title
    data['title_postags'] = data.titletoken_without_stopwords.apply(lambda x:count_postags(x))
    data['title_NN_count'] = data.title_postags.map(lambda x:x[0])
    data['title_VERB_count'] = data.title_postags.map(lambda x:x[1])
    data['title_ADJ_count'] = data.title_postags.map(lambda x:x[2])
    # text
    data['text_postags'] = data.texttoken_without_stopwords.apply(lambda x:count_postags(x))
    data['text_NN_count'] = data.text_postags.map(lambda x:x[0])
    data['text_VERB_count'] = data.text_postags.map(lambda x:x[1])
    data['text_ADJ_count'] = data.text_postags.map(lambda x:x[2])
    
    # create new features describe the length of the title and text, counting by words
    # also maybe try with counting by letters
    # and shall treat them as categorical variables group up with factor levels 
    data['title_length'] = data.title_token.apply(lambda x:len(x))
    data['text_length'] = data.text_token.apply(lambda x:len(x))
    
    return data

