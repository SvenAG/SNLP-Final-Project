import re
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, cross_validation, metrics 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

import cPickle as pickle	# for caching python objects


def main():
    data = map(json.loads, file('../data/html_extracted.json', 'r'))

    tags = ['title', 'h1', 'h2', 'h3', 'strong', 'b', 'a', 'img', 'meta_description', 'meta_keywords']
    
    for tag in tags:
        print "to do..."



if __name__ == "__main__":
    main()