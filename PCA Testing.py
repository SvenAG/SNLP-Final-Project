""" =================================================================
Code Harness for Kaggle Evergreen Competition
  Models using none of the raw data

For:
  CSCI-GA 3033 Statistical Natural Language Processing
  @ New York University
  Fall 2013
================================================================= """

import pandas as p
import numpy as np
import scipy as sp
import pylab as pl
from sklearn import linear_model, cross_validation, metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import decomposition
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
import pylab as pl
import random

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
modelType = "boilerplate_tfidf"         # choice between: "notext", "boilerplate_counter", "boilerplate_tfidf"
cv_folds = 10                           # number of cross validation folds
error_analysis = True                   # print confusion matrix

# ----------------------------------------------------------
# Prepare the Data
# ----------------------------------------------------------
training_data = np.array(p.read_table('../data/train.tsv'))
testing_data = np.array(p.read_table('../data/test.tsv'))

# 0 => "url"                       7 => "commonlinkratio_2"    14 => "hasDomainLink"       21 => "non_markup_alphanum_characters"
# 1 => "urlid"                     8 => "commonlinkratio_3"    15 => "html_ratio"          22 => "numberOfLinks"
# 2 => "boilerplate"               9 => "commonlinkratio_4"    16 => "image_ratio"         23 => "numwords_in_url
# 3 => "alchemy_category"         10 => "compression_ratio"    17 => "is_news"             24 => "parametrizedLinkRatio"
# 4 => "alchemy_category_score"   11 => "embed_ratio"          18 => "lengthyLinkDomain"   25 => "spelling_errors_ratio"
# 5 => "avglinksize"              12 => "framebased"           19 => "linkwordscore"       26 => "label"
# 6 => "commonlinkratio_1"        13 => "frameTagRatio"        20 => "news_front_page"

# get the target variable and set it as Y so we can predict it
Y = training_data[:,-1]

# not all data is numerical, so we'll have to convert those fields
# fix "is_news":
training_data[:,17] = [0 if x == "?" else 1 for x in training_data[:,17]]

# fix "news_front_page":
training_data[:,20] = [999 if x == "?" else x for x in training_data[:,20]]
training_data[:,20] = [1 if x == "1" else x for x in training_data[:,20]]
training_data[:,20] = [0 if x == "0" else x for x in training_data[:,20]]

# fix "alchemy category":
training_data[:,3] = [0 if x=="arts_entertainment" else x for x in training_data[:,3]]
training_data[:,3] = [1 if x=="business" else x for x in training_data[:,3]]
training_data[:,3] = [2 if x=="computer_internet" else x for x in training_data[:,3]]
training_data[:,3] = [3 if x=="culture_politics" else x for x in training_data[:,3]]
training_data[:,3] = [4 if x=="gaming" else x for x in training_data[:,3]]
training_data[:,3] = [5 if x=="health" else x for x in training_data[:,3]]
training_data[:,3] = [6 if x=="law_crime" else x for x in training_data[:,3]]
training_data[:,3] = [7 if x=="recreation" else x for x in training_data[:,3]]
training_data[:,3] = [8 if x=="religion" else x for x in training_data[:,3]]
training_data[:,3] = [9 if x=="science_technology" else x for x in training_data[:,3]]
training_data[:,3] = [10 if x=="sports" else x for x in training_data[:,3]]
training_data[:,3] = [11 if x=="unknown" else x for x in training_data[:,3]]
training_data[:,3] = [12 if x=="weather" else x for x in training_data[:,3]]
training_data[:,3] = [999 if x=="?" else x for x in training_data[:,3]]

# ----------------------------------------------------------
# Models
# ----------------------------------------------------------
if modelType == "notext":
    X = training_data[:,list([6, 8, 9, 19, 22, 25])]

    lr = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, class_weight=None, random_state=None)

elif modelType == "boilerplate_counter":
    X = training_data[:,2]

    counter = CountVectorizer(min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), stop_words=None)
    counter.fit(X)
    X = counter.transform(X)

    lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)

elif modelType == "boilerplate_tfidf":
    X = training_data[:,2]
    print(X.shape)
    print(Y.shape)
    #WHERE IS REMOVAL OF STOP WORDS!!??
    #LOWERCASE, NORMALIZE,
    #2 EXTRA FEATURES, NUMBER OF UNIQUE WORDS IN TITLE, BODY

    tfidf = TfidfVectorizer(min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfidf.fit(X)
    X = tfidf.transform(X)
    print(X.shape)          #7395 * 91629

    #sparsePCA = decomposition.SparsePCA(n_components=10000, alpha=0.8, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=1);
    #sparsePCA.fit(X)
    #X = sparsePCA.transform(X)
    #print(X.shape)

    truncatedSVD = decomposition.TruncatedSVD(n_components=100, algorithm='randomized', n_iterations=5)
    truncatedSVD.fit(X)
    X = truncatedSVD.transform(X)
    print(X.shape)
    i = input("lets see what we got here")
    lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)

print ("\nModel Type: ", modelType, "\nROC AUC: ", np.mean(cross_validation.cross_val_score(lr, X, Y, cv=cv_folds, scoring='roc_auc')))