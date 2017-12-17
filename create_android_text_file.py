import gzip

# get in form qid \t pid \w pid \t id \w id

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

if __name__ == "__main__":
	raw_corpus, all_sequences = read_corpus("../Android/corpus.tsv.gz")

	positives = {}
	negatives = {}

	print "loading data"
	# dev.[pos|neg].txt and test.[pos|neg].txt format:
	# id \w id
	pos_ids_X, pos_Y = load_data("../Android/test.pos.txt", True)
	for q1, q2 in pos_ids_X:
		if q1 in positives:
			positives[q1].append(q2)
		else:
			positives[q1] = [q2]

	neg_ids_X, neg_Y = load_data("../Android/test.neg.txt", False)
	for q1, q2 in neg_ids_X:
		if q1 in negatives:
			negatives[q1].append(q2)
		else:
			negatives[q1] = [q2]

	f = open('android_test.txt','w')
	question_ids = set()
	question_ids.update(positives.keys())
	question_ids.update(negatives.keys())

	for qid in question_ids:
		questions = qid + "\t"
		for pid in positives[qid]:
			questions += pid + " "
		questions = questions[:-1]
		questions += "\t"

		for pid in positives[qid]:
			questions += pid + " "

		for nid in negatives[qid]:
			questions += nid + " "
		questions = questions[:-1]
		questions += "\n"

		f.write(questions)

	f.close()
