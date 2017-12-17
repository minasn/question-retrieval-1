"""Part 2.3.1.a

Unsupervised methods used in the first part (Cosine Similarity etc).

Usage: python2 2a.py
"""

import gzip
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from meter import AUCMeter

def read_corpus(path):
    """Creates a dictionary mapping ID to a tuple
    tuple: dictionary for question title string, dictionary for body string"""
    print "reading corpus"
    raw_corpus = {}
    all_sequences = []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            title = title.lower().strip()
            body = body.lower().strip()
            raw_corpus[id] = (title, body)
            all_sequences.append(title.strip())
            all_sequences.append(body.strip())
    return raw_corpus, all_sequences

def load_data(filename, positive=True):
	"""Load question pairs with question id's.
	"""
	with open(filename, "rt") as f:
		X = []
		for line in f:
			X.append(line.strip().split(" "))
		y = [1 if positive else 0]*len(X)
	return X, y

# corpus.tsv format:
# id \t title \t body \n
raw_corpus, all_sequences = read_corpus("../Android/corpus.tsv.gz")

def calculate_meter(data):
	"""Calculate the AUC score.
	"""
	positives = {}
	negatives = {}

	print "loading data"
	# dev.[pos|neg].txt and test.[pos|neg].txt format:
	# id \w id
	if data == 'dev':
		pos_ids_X, pos_Y = load_data("../Android/dev.pos.txt", True)
	else:
		pos_ids_X, pos_Y = load_data("../Android/test.pos.txt", True)
	for q1, q2 in pos_ids_X:
		if q1 in positives:
			positives[q1].append(q2)
		else:
			positives[q1] = [q2]

	if data == 'dev':
		neg_ids_X, neg_Y = load_data("../Android/dev.neg.txt", False)
	else:
		neg_ids_X, neg_Y = load_data("../Android/test.neg.txt", False)
	for q1, q2 in neg_ids_X:
		if q1 in negatives:
			negatives[q1].append(q2)
		else:
			negatives[q1] = [q2]

	vectorizer = TfidfVectorizer()
	print "tfidf fit"
	vectorizer.fit(all_sequences)
	# 36404 unique words
	# print len(vectorizer.vocabulary_)

	meter = AUCMeter()

	qlabels = []
	all_questions = []
	question_ids = set()
	question_ids.update(positives.keys())
	question_ids.update(negatives.keys())
	for qid in question_ids:
		questions = [raw_corpus[qid][0] + " " + raw_corpus[qid][1]]
		questions.extend([raw_corpus[nid][0] + " " + raw_corpus[nid][1] for nid in negatives[qid]])
		questions.extend([raw_corpus[pid][0] + " " + raw_corpus[pid][1] for pid in positives[qid]])
		all_questions.append(questions)
		qlabels.append([0]*len(negatives[qid]) + [1]*len(positives[qid]))

	for question, qlabel in zip(all_questions, qlabels):
		query = torch.DoubleTensor(vectorizer.transform([question[0]]).todense())
		examples = torch.DoubleTensor(vectorizer.transform(question[1:]).todense())

		cos_similarity = F.cosine_similarity(query, examples, dim=1)
		target = torch.DoubleTensor(qlabel)
		meter.add(cos_similarity, target)

	print meter.value(0.05)

calculate_meter("dev")
calculate_meter("test")
