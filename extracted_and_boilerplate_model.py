""" =================================================================
Extracted and Boilerplate Models

For:
  CSCI-GA 3033 Statistical Natural Language Processing
  @ New York University
  Fall 2013
================================================================= """

import pandas as p
import numpy as np
import scipy as sp
#import pylab as pl

from sklearn import linear_model, ensemble, cross_validation, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *

import evergreen_tools


def main():
    # ----------------------------------------------------------
    # Settings
    # ----------------------------------------------------------
    data_type = "boilerplate"                # choice between: "extracted", "boilerplate"
    boilerplate_type = "tf-idf"             # choice between: "frequency", "tf-idf"
    model_type = "logistic_regression"
    cv_folds = 20                           # number of cross validation folds
    
    rfecv = False

    error_analysis = False

    # roc_plotter = False
    # roc_toFile = False
    # roc_filename = data_type + "_" + model_type + ".jpg"

    roc_data = False
    roc_data_filename = data_type + "_" + model_type + ".csv"

    leaderboard_output = False
    leaderboard_filename = data_type + "_" + model_type + ".csv"


    # ----------------------------------------------------------
    # Prepare the Data
    # ----------------------------------------------------------
    # import the data
    training_data = np.array(p.read_table('../data/train.tsv'))
    testing_data = np.array(p.read_table('../data/test.tsv'))
    
    Y = training_data[:, -1]    # get the outcome labels

    # length of training data
    training_length = training_data.shape[0]

    # combine the training and test so we can fit tfidfs
    all_data = np.vstack([training_data[:,0:26], testing_data])

    # 0 => "url"                       7 => "commonlinkratio_2"    14 => "hasDomainLink"       21 => "non_markup_alphanum_characters"
    # 1 => "urlid"                     8 => "commonlinkratio_3"    15 => "html_ratio"          22 => "numberOfLinks"
    # 2 => "boilerplate"               9 => "commonlinkratio_4"    16 => "image_ratio"         23 => "numwords_in_url
    # 3 => "alchemy_category"         10 => "compression_ratio"    17 => "is_news"             24 => "parametrizedLinkRatio"
    # 4 => "alchemy_category_score"   11 => "embed_ratio"          18 => "lengthyLinkDomain"   25 => "spelling_errors_ratio"
    # 5 => "avglinksize"              12 => "framebased"           19 => "linkwordscore"       26 => "label"
    # 6 => "commonlinkratio_1"        13 => "frameTagRatio"        20 => "news_front_page"

    # not all data is numerical, so we'll have to convert those fields
    
    # fix "is_news":
    all_data[:,17] = [0 if x == "?" else 1 for x in all_data[:,17]]

    # fix -1 entries in hasDomainLink
    all_data[:,14] = [0 if x =="-1" else x for x in all_data[:,10]]

    # fix "news_front_page":
    all_data[:,20] = [999 if x == "?" else x for x in all_data[:,20]]
    all_data[:,20] = [1 if x == "1" else x for x in all_data[:,20]]
    all_data[:,20] = [0 if x == "0" else x for x in all_data[:,20]]

    # fix "alchemy category":
    all_data[:,3] = [0 if x=="arts_entertainment" else x for x in all_data[:,3]]
    all_data[:,3] = [1 if x=="business" else x for x in all_data[:,3]]
    all_data[:,3] = [2 if x=="computer_internet" else x for x in all_data[:,3]]
    all_data[:,3] = [3 if x=="culture_politics" else x for x in all_data[:,3]]
    all_data[:,3] = [4 if x=="gaming" else x for x in all_data[:,3]]
    all_data[:,3] = [5 if x=="health" else x for x in all_data[:,3]]
    all_data[:,3] = [6 if x=="law_crime" else x for x in all_data[:,3]]
    all_data[:,3] = [7 if x=="recreation" else x for x in all_data[:,3]]
    all_data[:,3] = [8 if x=="religion" else x for x in all_data[:,3]]
    all_data[:,3] = [9 if x=="science_technology" else x for x in all_data[:,3]]
    all_data[:,3] = [10 if x=="sports" else x for x in all_data[:,3]]
    all_data[:,3] = [11 if x=="unknown" else x for x in all_data[:,3]]
    all_data[:,3] = [12 if x=="weather" else x for x in all_data[:,3]]
    all_data[:,3] = [999 if x=="?" else x for x in all_data[:,3]]

    all_features = list([3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    # ----------------------------------------------------------
    # Models
    # ----------------------------------------------------------
    if data_type == "extracted":
        X = all_data[0:training_length, all_features]
        X_test= all_data[training_length:, all_features]

        if model_type == "logistic_regression":
            m1 = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, class_weight=None, random_state=None)
        elif model_type == "random_forest":
            m1 = ensemble.RandomForestClassifier(n_estimators=1000,verbose=2,n_jobs=20,min_samples_split=5,random_state=1034324)

    elif data_type == "boilerplate":
        X_all = all_data[:,2]

        if boilerplate_type == "frequency":
            counter = CountVectorizer(min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), stop_words=None)
            counter.fit(X_all)
            X_all = counter.transform(X_all)
        elif boilerplate_type == "tf-idf":
            tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1)
            tfidf.fit(X_all)
            X_all = tfidf.transform(X_all)

        X = X_all[:training_length]
        X_test = X_all[training_length:]

        if model_type == "logistic_regression":
            m1 = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
        elif model_type == "random_forest":
            m1 = ensemble.RandomForestClassifier(n_estimators = 500)


    X = X.astype(float)
    Y = Y.astype(int)

    if rfecv:
        selector = RFECV(m1, step=1, cv=5)
        selector = selector.fit(X, Y)
        selected_cols = []
        selected_cols_original = []
        print selector.support_
        for s, b in enumerate(selector.support_):
            if b:
                selected_cols.append(s)
                selected_cols_original.append(all_features[s])

        X = X[:,list(selected_cols)]
        X_test = X_test[:,list(selected_cols)]

        print "Features before RFECV: " + str(all_features)
        print "Features after RFECV: " + str(selected_cols_original)


    X = X.astype(float)
    Y = Y.astype(int)

    print ("\nData Type: ", data_type, "\nModel Type: ", model_type, "\nROC AUC: ", np.mean(cross_validation.cross_val_score(m1, X, Y, cv=cv_folds, scoring='roc_auc')))

    # if roc_plotter:
    #     evergreen_tools.roc_plotter(m1, 10, False, roc_toFile, roc_filename, X, Y)

    if roc_data:
        evergreen_tools.roc_data(m1, X, Y, roc_data_filename)

    if leaderboard_output:
        evergreen_tools.leaderboard_ouput(m1, X, Y, X_test, testing_data[:,1], leaderboard_filename)


    # ----------------------------------------------------------
    # Errors Analysis - Confusion matrix
    # Does not use cross validation, but split the training set into train and test
    # ----------------------------------------------------------
    if error_analysis :
        
        tot_y_pred = []
        tot_y_test = []
        
        category_y_pred = [ [] for x in range(0,13) ]
        category_y_test = [ [] for x in range(0,13) ]
        
        # Kfold creates an iterator for each step in the cross validation
        cv = cross_validation.StratifiedKFold(Y, n_folds=cv_folds, indices=False) #iterates over all steps of the cross validation
        

        for train, test in cv:
            X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
            categories = all_data[test , 3]
            
            # question: can we fit and predict over different data using the same linear model? 
            # or should we copy the original linear model?
            y_pred = m1.fit(X_train, y_train).predict(X_test)
            for category in range(0,13) :
                index = np.where(categories == category)
                for ele in y_pred[index].tolist() : 
                    category_y_pred[category].append(ele)
                    tot_y_pred.append(ele)
                for ele in y_test[index].tolist() :
                    category_y_test[category].append(ele)
                    tot_y_test.append(ele)
        
        #prints confusion matrix for each category
        for category in range(0,13) :          
            y_test = np.array( category_y_pred[category] )
            y_pred = np.array( category_y_test[category] )
            y_test = map(int,y_test)
            y_pred = map(int,y_pred)
            cm = confusion_matrix(y_test, y_pred)
            print ("category = " , category )
            print (cm)

        # Compute confusion matrix for all data
        tot_y_test = np.array( tot_y_test )
        tot_y_pred = np.array( tot_y_pred )
        tot_y_test = map(int,tot_y_test)
        tot_y_pred = map(int,tot_y_pred)

        cm = confusion_matrix(tot_y_test, tot_y_pred)
        print ("all data confusion matrix")
        print (cm)

        # pretty-print
        pl.matshow(cm)
        pl.title('Confusion matrix')
        pl.colorbar()
        pl.ylabel('True label')
        pl.xlabel('Predicted label')
        pl.show()


if __name__ == "__main__":
    main()