import gzip
import numpy as np

import torch
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_corpus(path):
    """Creates a dictionary mapping ID to a tuple
    tuple: dictionary for question title string, dictionary for body string"""
    print "reading corpus"
    raw_corpus = {}
    all_words = set()
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            title = title.strip()
            body = body.strip()
            raw_corpus[id] = (title, body)
            all_words.update(title.strip().split(" "))
            all_words.update(body.strip().split(" "))
    return raw_corpus, all_words

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
raw_corpus, all_words = read_corpus("../Android/corpus.tsv.gz")

dev_positives = {}
dev_negatives = {}

print "loading data"
# dev.[pos|neg].txt and test.[pos|neg].txt format:
# id \w id
dev_pos_ids_X, dev_pos_Y = load_data("../Android/dev.pos.txt", True)
for q1, q2 in dev_pos_ids_X:
	if q1 in dev_positives:
		dev_positives[q1].append(q2)
	else:
		dev_positives[q1] = [q2]

dev_neg_ids_X, dev_neg_Y = load_data("../Android/dev.neg.txt", False)
# dev_neg_X = ids_to_text(dev_neg_ids_X, raw_corpus)
for q1, q2 in dev_neg_ids_X:
	if q1 in dev_negatives:
		dev_negatives[q1].append(q2)
	else:
		dev_negatives[q1] = [q2]

test_pos_ids_X, test_pos_Y = load_data("../Android/test.pos.txt", True)

test_neg_ids_X, test_neg_Y = load_data("../Android/test.neg.txt", False)

vectorizer = TfidfVectorizer()
print "tfidf fit transform"
tfidf_matrix = vectorizer.fit_transform(all_words)
# 36404 unique words
# print len(vectorizer.vocabulary_)

# dev
dev_qlabels = []
dev_questions = []
dev_question_ids = set()
dev_question_ids.update(dev_positives.keys())
dev_question_ids.update(dev_negatives.keys())
for qid in dev_question_ids:
	questions = [raw_corpus[qid][0] + " " + raw_corpus[qid][1]]
	questions.extend([raw_corpus[nid][0] + " " + raw_corpus[nid][1] for nid in dev_negatives[qid]])
	questions.extend([raw_corpus[pid][0] + " " + raw_corpus[pid][1] for pid in dev_positives[qid]])
	dev_questions.append(questions)
	dev_qlabels.extend([0]*len(dev_negatives[qid]))
	dev_qlabels.extend([1]*len(dev_positives[qid]))

print "transforming to tfidf"
# print dev_questions
dev_tfidf = [vectorizer.transform(dev_question) for dev_question in dev_questions]
dev_tfidf = np.array(dev_tfidf)

print dev_tfidf
print dev_tfidf.shape
print type(dev_tfidf[0])
print dev_tfidf[0].shape
dev_query = torch.DoubleTensor(dev_tfidf[0, :])
# dev_examples = [dev_example.todense() for dev_example in dev_tfidf[1:]]
# dev_examples = torch.DoubleTensor(np.array(dev_examples))

# print np.array(dev_tfidf)
print dev_query
# print dev_examples

quit()

print "calculating cosine similarities"
dev_cos_similarity = F.cosine_similarity(dev_query, dev_examples, dim=1)
dev_ranked_similarities = np.argsort(-1*dev_cos_similarity)
dev_positive_similarity = dev_qlabels[dev_ranked_similarities]

print "evaluating"
dev_evaluator = Evaluation(dev_positive_similarity)
print "precision at 1: " + str(dev_evaluator.Precision(1))
print "precision at 5: " + str(dev_evaluator.Precision(5))
print "MAP: " + str(dev_evaluator.MAP())
print "MRR: " + str(dev_evaluator.MRR())
