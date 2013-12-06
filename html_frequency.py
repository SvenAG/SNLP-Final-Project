import re
import json

import numpy as np
import pandas as p

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, cross_validation, metrics, ensemble
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
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
            lr.fit(X, Y)
            
            # get the score from the logistic regression for training and test
            if i == 0:
                scores = np.array(lr.predict_proba(X)[:,0])
                scores_test = np.array(lr.predict_proba(X_test)[:,0])
            else:
                scores = np.vstack((scores, lr.predict_proba(X)[:,0]))
                scores_test = np.vstack((scores_test, lr.predict_proba(X_test)[:,0]))


        # RANDOM FOREST
        if randomForestScores:
            # svd with 100 components on the tfidf output
            truncatedSVD = decomposition.TruncatedSVD(n_components=100, algorithm='randomized', n_iterations=5)  #Won't run for 1000
            truncatedSVD.fit(X_all)
            X_all = truncatedSVD.transform(X_all)

            # seperate training and test
            X = X_all[0:training_length]
            X_test = X_all[training_length:]

            # fit a random forest
            rf = ensemble.RandomForestClassifier(n_estimators = 500)
            rf.fit(X,Y)

            # get the scores of the random forest for training and test
            if i == 0 and not logisticRegressionScores:
                scores = np.array(rf.predict_proba(X)[:,0])
                scores_test = np.array(rf.predict_proba(X_test)[:,0])
            else:
                scores = np.vstack((scores, rf.predict_proba(X)[:,0]))
                scores_test = np.vstack((scores_test, rf.predict_proba(X_test)[:,0]))

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
    print "Model: " + "Logistic Regression" + "\tROC AUC: ", np.mean(cross_validation.cross_val_score(lr, X, Y, cv=10, scoring='roc_auc'))
    # END COMBINE MODELS ###################################################################


    if leaderboard_output:
        # fit a logistic regression with the scores and output labels
        lr.fit(X, Y)

        # predict the test data
        prediction = lr.predict(X_test)

        # output will be url_id, predicition
        output = np.array(testing_data[:,1])
        output = np.vstack((output, prediction)).T
        predictions = p.DataFrame(data=output, columns=['urlid', 'label'])

        # save the output
        predictions.to_csv('../data/output.csv', index=False)


if __name__ == "__main__":
    main()