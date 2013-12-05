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

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
import pylab as pl
import random

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
modelType = "kMeans"         # choice between: "notext", "boilerplate_counter", "boilerplate_tfidf"
cv_folds = 10                           # number of cross validation folds
error_analysis = True                   # print confusion matrix

# ----------------------------------------------------------
# Prepare the Data
# ----------------------------------------------------------
training_data = np.array(p.read_table('../data/train.csv'))
testing_data = np.array(p.read_table('../data/test.csv'))

# 0 => "url"                       7 => "commonlinkratio_2"    14 => "hasDomainLink"       21 => "non_markup_alphanum_characters"
# 1 => "urlid"                     8 => "commonlinkratio_3"    15 => "html_ratio"          22 => "numberOfLinks"
# 2 => "boilerplate"               9 => "commonlinkratio_4"    16 => "image_ratio"         23 => "numwords_in_url
# 3 => "alchemy_category"         10 => "compression_ratio"    17 => "is_news"             24 => "parametrizedLinkRatio"
# 4 => "alchemy_category_score"   11 => "embed_ratio"          18 => "lengthyLinkDomain"   25 => "spelling_errors_ratio"
# 5 => "avglinksize"              12 => "framebased"           19 => "linkwordscore"       26 => "label"
# 6 => "commonlinkratio_1"        13 => "frameTagRatio"        20 => "news_front_page"

# get the target variable and set it as Y so we can predict it
# Y = training_data[:,-1]

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

for category in range(0,13) :
    w = list(np.where( training_data[:,3] == category )[0])
    training_data_temp = training_data
    training_data = training_data[w,:]
    Y = training_data[:,-1]
    
    # print training_data
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
        tfidf = TfidfVectorizer(min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1)
        tfidf.fit(X)
        X = tfidf.transform(X)

        lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)

    elif modelType == "kMeans":
        X = training_data[:,2]
        tfidf = TfidfVectorizer(min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1)
        tfidf.fit(X)
        X = tfidf.transform(X)

        labeler = kMeans(k=10)
        labeler.fit(X.tocsr())

        for (row, label) in enumerate(labeler.labels_):   
            print "row %d has label %d"%(row, label)  

    print ("\nModel Type: ", modelType, "\nROC AUC: ", np.mean(cross_validation.cross_val_score(lr, X, Y, cv=cv_folds, scoring='roc_auc')))
    print ("\nModel Type: ", modelType, "\nROC AUC: ", cross_validation.cross_val_score(lr, X, Y, cv=cv_folds, scoring='roc_auc'))

    # ----------------------------------------------------------
    # Errors Analysis - Confusion matrix
    # Does not use cross validation, but split the training set into train and test
    # ----------------------------------------------------------

    if error_analysis :
        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y ,  test_size= (1.0/cv_folds),  random_state=0)
        
        # Run classifier
        y_pred = lr.fit(X_train, y_train).predict(X_test)
        y_test = map(int,y_test)
        y_pred = map(int,y_pred)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print cm

        #pretty-print
        # pl.matshow(cm)
        # pl.title('Confusion matrix')
        # pl.colorbar()
        # pl.ylabel('True label')
        # pl.xlabel('Predicted label')
        # pl.show()
        
        #confusion matrix separeted by category
        training_data = training_data_temp 
    