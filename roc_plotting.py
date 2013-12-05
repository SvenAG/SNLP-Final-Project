""" =================================================================
Code Harness for Kaggle Evergreen Competition
  Models using none of the raw data

For:
  CSCI-GA 3033 Statistical Natural Language Processing
  @ New York University
  Fall 2013
================================================================= """

import pandas as p
import numpy as npy
import scipy
import pylab as pl
from sklearn import linear_model, cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
import pylab as pl
import random

def main():
    # ----------------------------------------------------------
    # Settings
    # ----------------------------------------------------------
    modelType = "boilerplate_tfidf"         # choice between: "notext", "boilerplate_counter", "boilerplate_tfidf"
    cv_folds = 10                           # number of cross validation folds
    error_analysis = True                   # print confusion matrix
    leaderboard_test = True                 # generate output for the leaderboard

    # ----------------------------------------------------------
    # Prepare the Data
    # ----------------------------------------------------------
    training_data = npy.array(p.read_table('../data/train.tsv'))
    testing_data = npy.array(p.read_table('../data/test.tsv'))

    all_data = npy.vstack([training_data[:,0:26], testing_data])

    training_length = training_data.shape[0]

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

    Y = training_data[:,-1].astype(int)

    if modelType == "notext":
        X = training_data[:,list([6, 8, 9, 19, 22, 25])]
        X_test = training_data[:,list([6, 8, 9, 19, 22, 25])]

        lr = linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, class_weight=None, random_state=None)

    elif modelType == "boilerplate_counter":
        X_all = all_data[:,2]

        counter = CountVectorizer(min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), stop_words=None)
        counter.fit(X_all)
        X_all = counter.transform(X_all)

        X = X_all[0:training_length]
        X_test = X_all[training_length:]

        lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)

    elif modelType == "boilerplate_tfidf":
        X_all = all_data[:,2]

        tfidf = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)
        tfidf.fit(X_all)
        X_all = tfidf.transform(X_all)

        X = X_all[0:training_length]
        X_test = X_all[training_length:]

        lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)

    print "\nModel Type: ", modelType, "\nROC AUC: ", npy.mean(cross_validation.cross_val_score(lr, X, Y, cv=cv_folds, scoring='roc_auc'))
        #print "\nModel Type: ", modelType, "\nROC AUC: ", cross_validation.cross_val_score(lr, X, Y, cv=cv_folds, scoring='roc_auc')
    roc_plotter(X, Y)

    if leaderboard_test:
        lr.fit(X, Y)
        prediction = lr.predict(X_test)
        output = npy.array(testing_data[:,1])
        output = npy.vstack((output, prediction)).T
        predictions = p.DataFrame(data=output, columns=['urlid', 'label'])
        predictions.to_csv('../data/output.csv', index=False)

def logreg(X_set, Y_set):
    mean_tpr = 0.0
    mean_fpr = npy.linspace(0, 1, 100)
    
    cv = cross_validation.KFold(len(Y_set), n_folds=5)
    X_set = X_set.astype(float)
    for i, (train, test) in enumerate(cv):
        print "Fold-%d:" % (i+1)
        
        # fit Logistic Regression model
        m1 = linear_model.LogisticRegression()
        m1.fit(X_set[train], Y_set[train])
        probas = m1.predict_proba(X_set[test])
        fpr, tpr, thresholds = metrics.roc_curve(Y_set[test], probas[:,1])
        roc_auc = metrics.auc(fpr, tpr)
        print "LR: AUC=%.6f" % roc_auc
            
        mean_tpr += scipy.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
        w = npy.where(npy.array(Y_set[test] == True))
    
    

    mean_tpr /= len(cv)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)*100
    pl.plot(mean_fpr, mean_tpr, 'k-', color=(npy.random.random(1)[0], npy.random.random(1)[0], npy.random.random(1)[0]), label='ROC (area = %0.2f)' % mean_auc, lw=2)
    
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    pl.show()
    

if __name__ == "__main__":
    main()