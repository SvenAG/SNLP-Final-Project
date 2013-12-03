import re
import json

import numpy as np
import pandas as p

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, cross_validation, metrics 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

import cPickle as pickle	# for caching python objects


def main():
	training_data = np.array(p.read_table('../data/train.tsv'))
	testing_data = np.array(p.read_table('../data/test.tsv'))

	training_length = training_data.shape[0]

	all_data = np.vstack([training_data[:,0:26], testing_data])


	data = map(json.loads, file('../data/html_extracted.json', 'r'))

	tags = ['title', 'h1', 'h2', 'h3', 'strong', 'b', 'a', 'img', 'meta_description', 'meta_keywords', 'boilerplate', 'summary']

	extracted = np.array(all_data[:, 1])

	print "Extracting tags..."

	for i, tag in enumerate(tags):
		if tag != "lolol":
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
	models = {}

	for i, tag in enumerate(tags):
		if tag != "lolol":
			col = i + 1
			X = extracted[0:training_length, col]
			Y = training_data[:, 26]
			tfidf = TfidfVectorizer(min_df=1, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1)
			tfidf.fit(X)
			X = tfidf.transform(X)

			models[tag] = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, class_weight=None, random_state=None)
			print "\nModel " + str(i+1) + "/" + str(len(tags)) + " trained."
			print "Tag: " + tag + "\tROC AUC: ", np.mean(cross_validation.cross_val_score(models[tag], X, Y, cv=10, scoring='roc_auc'))

	print "Training done."

if __name__ == "__main__":
    main()