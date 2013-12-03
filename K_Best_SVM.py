import sklearn
import pandas as p
import numpy as np
import scipy as sp
import pylab as pl
from sklearn import linear_model, cross_validation, metrics
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
from numpy import array
from numpy import savetxt
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_selection import SelectKBest, chi2


modelType = "notext"


# ----------------------------------------------------------
# Prepare the Data
# ----------------------------------------------------------
training_data = np.array(p.read_table('../data/train.tsv'))
print ("Read Data\n")

# get the target variable and set it as Y so we can predict it
Y = training_data[:,-1].astype(int)

print(Y)

# not all data is numerical, so we'll have to convert those fields
# fix "is_news":
training_data[:,17] = [0 if x == "?" else 1 for x in training_data[:,17]]

# fix -1 entries in hasDomainLink
training_data[:,14] = [0 if x =="-1" else x for x in training_data[:,10]]

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

print ("Corrected outliers data\n")

# ----------------------------------------------------------
# Models
# ----------------------------------------------------------
if modelType == "notext":
    print ("no text model\n")
    #ignore features which are useless
    X = training_data[:,list([3, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 19, 20, 22, 25])]
#    X = array([x[1:] for x in X])
#    Y = array(Y)
    scaler = preprocessing.StandardScaler()
    print("initialized scaler \n")
    scaler.fit(X,Y)
    print("fitted train data and labels\n")
    X = scaler.transform(X)
    print("Transformed train data\n")
    svc = SVC(kernel = "linear")
    print("Initialized SVM\n")
    rfecv = RFECV(estimator = svc, cv = 5, step=1, verbose = 1)
    print("Initialized RFECV\n")
    X = rfecv.fit(X,Y)
#    print("Fitted train data and label\n")
#    rfecv.support_
    print ("Optimal Number of features : %d" % rfecv.n_features_)
    savetxt('rfecv.csv', rfecv.ranking_, delimiter=',', fmt='%f')
#    savetxt('transformedtrain.tsv', X, delimiter=',', fmt='%f')
    exit()
