import re
import json

import numpy as np
import scipy
import pandas as p
import pylab as pl

from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model, cross_validation, metrics, ensemble
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFECV
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfTransformer

import cPickle as pickle    # for caching python objects


def main():
    # SETTINGS ###################################################################
    leaderboard_output = True
    logisticRegressionScores = True
    randomForestScores = False

    # which fields from the html data do we care about?
    tags = ['url', 'title', 'h1', 'h2', 'h3', 'strong', 'b', 'a', 'img', 'meta_description', 'meta_keywords', 'boilerplate', 'summary']
    #tags = ['url']
    # END SETTINGS ###################################################################



    # DATA IMPORT ###################################################################
    # import the data
    training_data = np.array(p.read_table('../data/train.tsv'))
    testing_data = np.array(p.read_table('../data/test.tsv'))
    
    Y = training_data[:, 26]    # get the outcome labels

    # length of training data
    training_length = training_data.shape[0]

    # combine the training and test so we can fit tfidfs
    all_data = np.vstack([training_data[:,0:26], testing_data])

    # read in the html data using json
    data = map(json.loads, file('../data/html_extracted.json', 'r'))

    # we will extract the data from the json entries for each URL

    # make the first column of extracted the url_ids
    extracted = np.array(all_data[:, 1])

    print "Extracting tags..."

    # go through each tag/field we care about
    for i, tag in enumerate(tags):
        incoming = []                                   # the combined output of every url's tag will go in here
        for datum in data:                              # go through each tag in the url entry
            text = ""                                   # the combined text from each url's tag will go here
            if tag in datum:                            # do we have anything for this url-tag combination?
                for element in datum[tag]:              # this will get all text is the tag occured more than once (i.e. many links, h1s, etc)
                    text += element + " "               # add it along with a space to seperate the next one
            incoming.append(text)                       # add it to the output
        extracted = np.vstack((extracted, incoming))    #add the tag to the extracted matrix
        
        print "Tag " + str(i+1) + "/" + str(len(tags)) + " extracted."

    extracted = extracted.T     # we have to flip the output matrix
    
    print "Extraction done."

    print "\nShape of extraction: " + str(extracted.shape) + "\n"
    # END DATA IMPORT ###################################################################



    # MODELS ###################################################################
    # we will train some models for each tag
    print "Training models..."
    
    # go through all the tags we care about
    for i, tag in enumerate(tags):
        col = i + 1     # col is the position the tag is in the extracted matrix
        
        X_all = extracted[:, col]   # get the text for this tag
        
        # fit a tfidf on all the text for that tag
        tfidf = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)
        tfidf.fit(X_all)
        X_all = tfidf.transform(X_all)

        # seperate training and test
        X = X_all[0:training_length]
        X_test = X_all[training_length:]


        # LOGISTIC REGRESSION
        if logisticRegressionScores:
            # fit a logistic regression for this tag
            lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)
            
            # svd with 100 components on the tfidf output
            # truncatedSVD = decomposition.TruncatedSVD(n_components=100, algorithm='randomized', n_iterations=5)  #Won't run for 1000
            # truncatedSVD.fit(X_all)
            # X_all = truncatedSVD.transform(X_all)

            # # seperate training and test
            # X = X_all[0:training_length]
            # X_test = X_all[training_length:]

            X = X.astype(float)
            Y = Y.astype(int)

            # X_before_gs = np.array(X, copy=True)
            # X_test_before_gs = np.array(X_test, copy=True)

            # print "Shape of X before GS: " + str(X.shape)

            # selector = RFECV(lr, step=1, cv=5)
            # selector = selector.fit(X, Y)
            # selected_cols = []
            # for s, b in enumerate(selector.support_):
            #     if b:
            #         selected_cols.append(s)

            # X = X[:,list(selected_cols)]
            # X_test = X_test[:,list(selected_cols)]
            
            # print "Shape of X after GS: " + str(X.shape)

            lr.fit(X, Y)
            
            # get the score from the logistic regression for training and test
            if i == 0:
                # scores = np.array(lr.predict_proba(X)[:,0])
                # scores_test = np.array(lr.predict_proba(X_test)[:,0])
                scores = np.array(lr.predict(X))
                scores_test = np.array(lr.predict(X_test))
            else:
                # scores = np.vstack((scores, lr.predict_proba(X)[:,0]))
                # scores_test = np.vstack((scores_test, lr.predict_proba(X_test)[:,0]))
                scores = np.vstack((scores, lr.predict(X)))
                scores_test = np.vstack((scores_test, lr.predict(X_test)))


        # RANDOM FOREST
        if randomForestScores:
            # X = X_before_gs
            # X_test = X_test_before_gs

            rf = ensemble.RandomForestClassifier(n_estimators = 500)

            # svd with 100 components on the tfidf output
            truncatedSVD = decomposition.TruncatedSVD(n_components=100, algorithm='randomized', n_iterations=5)  #Won't run for 1000
            truncatedSVD.fit(X_all)
            X_all = truncatedSVD.transform(X_all)

            # seperate training and test
            X = X_all[0:training_length]
            X_test = X_all[training_length:]

            X = X.astype(float)
            Y = Y.astype(int)

            # print "Shape of X before GS: " + str(X.shape)

            # selector = RFECV(rf, step=1, cv=5)
            # selector = selector.fit(X, Y)
            # selected_cols = []
            # for s, b in enumerate(selector.support_):
            #     if b:
            #         selected_cols.append(s)

            # X = X[:,list(selected_cols)]
            # X_test = X_test[:,list(selected_cols)]

            # print "Shape of X after GS: " + str(X.shape)

            # fit a random forest
            rf.fit(X,Y)

            # get the scores of the random forest for training and test
            if i == 0 and not logisticRegressionScores:
                # scores = np.array(rf.predict_proba(X)[:,0])
                # scores_test = np.array(rf.predict_proba(X_test)[:,0])
                scores = np.array(rf.predict(X))
                scores_test = np.array(rf.predict(X_test))
            else:
                # scores = np.vstack((scores, rf.predict_proba(X)[:,0]))
                # scores_test = np.vstack((scores_test, rf.predict_proba(X_test)[:,0]))
                scores = np.vstack((scores, rf.predict(X)))
                scores_test = np.vstack((scores_test, rf.predict(X_test)))

        print "Model " + str(i+1) + "/" + str(len(tags)) + " trained."
        print "Tag: " + tag + "\tROC AUC: ", str(np.mean(cross_validation.cross_val_score(lr, X, Y, cv=10, scoring='roc_auc')))

    # flip the scores so we have a row for each URL and a column for each score (# rows = # urls, # cols = # tags * 2(lr and rf))
    X = scores.T
    X_test = scores_test.T
    
    print "\nShape of model scores: " + str(scores.shape)
    
    print "Training done."
    # END MODELS ###################################################################


    # COMBINE MODELS ###################################################################
    print "\nUsing tag-models as features in new models..."

    # make a new logistic regression
    lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)
    roc_plotter(X, Y)
    print "Model: " + "Logistic Regression" + "\tROC AUC: ", np.mean(cross_validation.cross_val_score(lr, X, Y, cv=10, scoring='roc_auc'))
    # END COMBINE MODELS ###################################################################


    if leaderboard_output:
        # fit a logistic regression with the scores and output labels
        X = X.astype(float)
        Y = Y.astype(int)

        selector = RFECV(lr, step=1, cv=5)
        selector = selector.fit(X, Y)
        selected_cols = []
        for s, b in enumerate(selector.support_):
            if b:
                selected_cols.append(s)

        X = X[:,list(selected_cols)]
        X_test = X_test[:,list(selected_cols)]

        lr.fit(X, Y)

        # predict the test data
        prediction = lr.predict(X_test)

        # output will be url_id, predicition
        output = np.array(testing_data[:,1])
        output = np.vstack((output, prediction)).T
        predictions = p.DataFrame(data=output, columns=['urlid', 'label'])

        # save the output
        predictions.to_csv('../data/output_Rlr_both100.csv', index=False)

def roc_plotter(X_set, Y_set):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    cv = cross_validation.KFold(len(Y_set), n_folds=10)
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
        
        w = np.where(np.array(Y_set[test] == True))
    
    

    mean_tpr /= len(cv)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)*100
    pl.plot(mean_fpr, mean_tpr, 'k-', label='ROC (area = %0.2f)' % mean_auc, lw=2)
    
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