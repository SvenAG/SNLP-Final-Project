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
import scipy as sp
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import decomposition
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans 
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
import random

import codecs

import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    output_train = codecs.open('train_new_categories_12','w','utf-8')
    output_test = codecs.open('test_new_categories_12','w','utf-8')
    # ----------------------------------------------------------
    # Settings
    # ----------------------------------------------------------
    modelType = "kMeans"         # choice between: "notext", "boilerplate_counter", "boilerplate_tfidf"
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

    X = training_data[0:,2]
    Z = testing_data[0:,2]

    print "TRAINING DATA EXTRACTED!"

    tfidf = TfidfVectorizer( min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfidf.fit(X)
    X = tfidf.transform(X)

    print "TF-IDF PERFORMED!"

    lsa = TruncatedSVD(100)
    X = lsa.fit_transform(X)
    X = Normalizer(copy=False).fit_transform(X)

    km = KMeans(n_clusters=12, init='k-means++', max_iter=100, n_init=1,
        verbose=True)

    km.fit(X)

    print "CLUSTERING PERFORMED!"

    label_m = [[] for i in range(15)]
    print len(label_m)

    for (row, label) in enumerate(km.labels_):   
        print "row %d has label %d"%(row, label) 
        label_m[label].append(row)

        output_train.write(str(label)+'\n')


    output_train.close()

    Z = tfidf.transform(Z)
    Z = lsa.fit_transform(Z)
    Z = Normalizer(copy=False).fit_transform(Z)   

    for i in km.predict(Z):
        output_test.write(str(i)+'\n')
        

##############  LDA ###############
    # elif modelType == "LDA":
    #     stoplist = set('for a of the and to in'.split())
    #     X_all = all_data[:,2]

    #     # X_all = []

    #     # for element in X_dat:
    #     #     row = ''
    #     #     try:
    #     #         for key in eval(element).keys():
    #     #             row += (eval(element)[key])
    #     #         X_all.append(row.lower())
    #     #     except:
    #     #         pass
    #     #     try:
    #     #         for key in eval(element).keys():
    #     #             if key not in keys:
    #     #                 keys.add(key)
    #     #     except:
    #     #         pass
    #     #     # X_all.append(eval(element)['title']+' '+eval(element)['url']+' '+eval(element)['body'])
            
    #     # print keys
    #     # print X_all[0]
    #     # exit()

    #     id2word = gensim.corpora.Dictionary.load_from_text(X_all)
    #     mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
    #     print mm
    #     exit()
    #     print 'test'
    #     tokens = []

    #     texts = [[word for word in document.lower().split()] for document in X_all]

    #     print 'lower'

    #     all_tokens = sum(texts, [])
    #     print 'sums'
    #     tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

    #     texts = [[word for word in text if word not in tokens_once]
    #           for text in X_all]

    #     print 'TOKENIZED'

    #     dictionary = corpora.Dictionary(texts)
    #     print 'DICTIONARY CREATED'

    #     tfidf = TfidfVectorizer(stop_words = 'english', min_df=2, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)
    #     tfidf.fit(X_all)
    #     X_all = tfidf.transform(X_all)

    #     "TFIDF PERFORMED"

    #     X = X_all[0:training_length]
    #     X_test = X_all[training_length:]

    #     # corpus_tfidf = tfidf[corpus]

    #     lsi = models.LdaModel(X, id2word=dictionary, num_topics=2)#models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    #     sims = index[lsi]
    #     print list(enumerate(sims))
    #     # truncatedSVD = decomposition.TruncatedSVD(n_components=1000, algorithm='randomized', n_iterations=5)  #Won't run for 1000
    #     # truncatedSVD.fit(X)
    #     # X = truncatedSVD.transform(X)
    #     # print(X.shape)

    #     # #X = X.toarray()

    #     #corpus = gensim.matutils.Sparse2Corpus(X)

    

if __name__ == "__main__":
    main()