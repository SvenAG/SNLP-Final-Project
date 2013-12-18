""" =================================================================
Evergreen Tools

FUNCTION roc_plotter:
 - Expects to be given a model, # folds, boolean to print fold output, 
   an X-matrix, and a Y-matrix

FUNCTION roc_data:
 - Expects a model, X-matrix, Y-matrix, and output filename

FUNCTION leaderboard_output:
 - Expects to be given a model, X-matrix, Y-matrix, X_test-matrix, the
   urlids for the output, boolean to use rfecv, the filename

For:
  CSCI-GA 3033 Statistical Natural Language Processing
  @ New York University
  Fall 2013
================================================================= """

import numpy as np
import scipy as sp
import pandas as p
#import pylab as pl
import random

from sklearn import linear_model, cross_validation, metrics
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.metrics import roc_curve, auc, metrics


# def roc_plotter(m1, folds, verbose, toFile, filename, X, Y):
#     mean_tpr = 0.0
#     mean_fpr = np.linspace(0, 1, 100)

#     cv = cross_validation.StratifiedKFold(Y, n_folds=folds)
#     X = X.astype(float)
#     for i, (train, test) in enumerate(cv):
        

#         # fit the model
#         m1.fit(X[train], Y[train])

#         # predict
#         probas = m1.predict_proba(X[test])
#         fpr, tpr, thresholds = metrics.roc_curve(Y[test], probas[:,1])
#         roc_auc = metrics.auc(fpr, tpr)
        
#         if verbose:
#             print ("Fold-%d:" % (i+1))
#             print ("LR: AUC=%.6f" % roc_auc)

#         mean_tpr += sp.interp(mean_fpr, fpr, tpr)
#         mean_tpr[0] = 0.0

#         w = np.where(np.array(Y[test] == True))

#     mean_tpr /= len(cv)
#     mean_auc = metrics.auc(mean_fpr, mean_tpr)*100
#     pl.plot(mean_fpr, mean_tpr, 'k-', label='ROC (area = %0.2f)' % mean_auc, lw=2)

#     pl.xlim([-0.05, 1.05])
#     pl.ylim([-0.05, 1.05])
#     pl.xlabel('False Positive Rate')
#     pl.ylabel('True Positive Rate')
#     pl.title('Receiver Operating Characteristic')
#     pl.legend(loc="lower right")
#     pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
#     pl.show()


def roc_data(m1, X, Y, filename):
    X = X.astype(float)
    Y = Y.astype(int)

    s = np.zeros(len(Y))

    for train_idx, test_idx in cross_validation.KFold(len(Y), 10):
        m1.fit(X[train_idx], Y[train_idx])
        
        s[list(test_idx)] = m1.predict_proba(X[test_idx])[:, -1]

    output = np.vstack((Y, s)).T
    roc_datum = p.DataFrame(data=output, columns=['label', 'probability'])
    roc_datum.to_csv('roc_plotting/' + filename, index=False)


def leaderboard_ouput(m1, X, Y, X_test, urlids, filename):
    X = X.astype(float)
    X_test = X_test.astype(float)
    Y = Y.astype(int)

    m1.fit(X, Y)
    prediction = m1.predict_proba(X_test)[:,1]
    output = np.array(urlids)
    output = np.vstack((output, prediction)).T
    predictions = p.DataFrame(data=output, columns=['urlid', 'label'])
    predictions.to_csv('../data/' + filename, index=False)