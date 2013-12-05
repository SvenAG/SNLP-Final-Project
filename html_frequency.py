import re
import json

import numpy as np
import pandas as p

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, cross_validation, metrics, ensemble
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

import cPickle as pickle    # for caching python objects


def main():
    training_data = np.array(p.read_table('../data/train.tsv'))
    testing_data = np.array(p.read_table('../data/test.tsv'))

    training_length = training_data.shape[0]

    all_data = np.vstack([training_data[:,0:26], testing_data])


    data = map(json.loads, file('../data/html_extracted_p.json', 'r'))

    #tags = ['title', 'h1', 'h2', 'h3', 'strong', 'b', 'a', 'img', 'meta_description', 'meta_keywords', 'boilerplate', 'summary']
    tags = ['h1', 'h2']
    #tags = ['title', 'meta_keywords', 'summary', 'boilerplate']

    extracted = np.array(all_data[:, 1])

    print "Extracting tags..."

    for i, tag in enumerate(tags):
        if tag != "lol":
            incoming = []
            for datum in data:
                text = ""
                if tag in datum:
                    for element in datum[tag]:
                        text += element + " "
                incoming.append(text)
            extracted = np.vstack((extracted, incoming))
            print "Tag " + str(i+1) + "/" + str(len(tags)) + " extracted."

    extracted = extracted.T
    print "Extraction done."

    
    print "\nShape of extraction: " + str(extracted.shape) + "\n"
    #print extracted[1:4, :]

    print "Training models..."
    
    for i, tag in enumerate(tags):
        if tag != "lol":

            col = i + 1
            X_all = extracted[:, col]
            Y = training_data[:, 26]
            tfidf = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)
            #tfidf.fit(X_all)
            tfidf.fit(extracted[0:training_length, col])
            X_all = tfidf.transform(X_all)

            X = X_all[0:training_length]
            X_test = X_all[training_length:]

            lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)
            #lr = ensemble.RandomForestClassifier()
            print "Model " + str(i+1) + "/" + str(len(tags)) + " trained."
            print "Tag: " + tag + "\tROC AUC: ", str(np.mean(cross_validation.cross_val_score(lr, X, Y, cv=10, scoring='roc_auc')))

            if col != len(tags):
            	print 

            lr.fit(X, Y)
            
            #print X.shape
            #print Y.shape
            
            if i == 0:
                scores = np.array(lr.predict_proba(X)[:,0])
                scores_test = np.array(lr.predict_proba(X_test)[:,0])
            else:
                scores = np.vstack((scores, lr.predict_proba(X)[:,0]))
                scores_test = np.vstack((scores_test, lr.predict_proba(X_test)[:,0]))
            #scores[tag] = score[:,0]
            #print score[:,0]

    X = scores.T
    X_test = scores_test.T
    print "\nShape of model scores: " + str(scores.shape)
    
    print "Training done."

    print "\nUsing tag-models as features in new models..."

    Y = training_data[:, 26]
    lr = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)
    print "Model: " + "Logistic Regression" + "\tROC AUC: ", np.mean(cross_validation.cross_val_score(lr, X, Y, cv=10, scoring='roc_auc'))

    predict = True

    if predict:
        lr.fit(X, Y)
        prediction = lr.predict(X_test)
        output = np.array(testing_data[:,1])
        output = np.vstack((output, prediction)).T
        predictions = p.DataFrame(data=output, columns=['urlid', 'label'])
        predictions.to_csv('../data/output.csv', index=False)


if __name__ == "__main__":
    main()