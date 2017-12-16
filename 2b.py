import sys
import os
import argparse

import numpy as np
import csv

import corpus
from evaluation import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam

def main(args):
    cnn = load_model(args)
    print "loaded cnn model"

    raw_corpus = corpus.read_corpus(args.corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    print("loaded embeddings")
    ids_corpus = corpus.map_corpus(vocab_map, raw_corpus)

    evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, cnn)

def load_model(args):
    print("loading " + args.load_model)
    cnn = nn.Conv1d(in_channels=200, out_channels=args.hidden_size, kernel_size = 3, padding = 1)
    cnn.load_state_dict(torch.load(args.load_model))
    optimizer = Adam(cnn.parameters())
    if args.cuda:
        cnn.cuda()

def evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, cnn):
    print "starting evaluation"
    val_data = corpus.read_annotations(args.test)
    print "number of lines in test data: " + str(len(val_data))
    val_batches = corpus.create_eval_batches(ids_corpus, val_data, padding_id)
    count = 0
    similarities = []

    for batch in val_batches:
        titles, bodies, qlabels = batch
        title_length, title_num_questions = titles.shape
        body_length, body_num_questions = bodies.shape
        title_embeddings, body_embeddings = corpus.get_embeddings(titles, bodies, vocab_map, embeddings)
        
        if args.cuda:
            title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings).cuda())]
        else:
            title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
        title_inputs = torch.cat(title_inputs).transpose(0,1).transpose(1,2)

        title_out = cnn(title_inputs)
        title_out = F.tanh(title_out)
        title_out = title_out.transpose(1,2).transpose(0,1)

        average_title_out = average_questions(title_out, titles, padding_id)

        # body
        if args.cuda:
            body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings).cuda())]
        else:
            body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
        body_inputs = torch.cat(body_inputs).transpose(0,1).transpose(1,2)
        
        body_out = cnn(body_inputs)
        body_out = F.tanh(body_out)
        body_out = body_out.transpose(1,2).transpose(0,1)

        # average all words of each question from body_out
        average_body_out = average_questions(body_out, bodies, padding_id)

        # average body and title
        # representations of the questions as found by the CNN
        # 560 x 100
        hidden = (average_title_out + average_body_out) * 0.5

        query = hidden[0].unsqueeze(0)
        examples = hidden[1:]

        cos_similarity = F.cosine_similarity(query, examples, dim=1)
        cos_similarity_np = cos_similarity.cpu().data.numpy()
        ranked_similarities = np.argsort(-1*cos_similarity_np)
        positive_similarity = qlabels[ranked_similarities]
        similarities.append(positive_similarity)

    evaluator = Evaluation(similarities)
    metrics = [evaluator.MAP(), evaluator.MRR(), str(evaluator.Precision(1)), str(evaluator.Precision(5))]
    print "precision at 1: " + str(evaluator.Precision(1))
    print "precision at 5: " + str(evaluator.Precision(5))
    print "MAP: " + str(evaluator.MAP())
    print "MRR: " + str(evaluator.MRR())

    with open(os.path.join(sys.path[0],args.results_file), 'a') as evaluate_file:
        writer = csv.writer(evaluate_file, dialect='excel')
        writer.writerow(metrics)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus",
            type = str
        )
    argparser.add_argument("--test",
            type = str,
            default = ""
        )
    argparser.add_argument("--embeddings",
            type = str,
            default = ""
        )
    argparser.add_argument("--hidden_size",
            type = int,
            default = 100
        )
    argparser.add_argument("--cuda",
            type = int,
            default = 0
        )
    argparser.add_argument("--load_model",
            type = str,
            default = ""
        )
    argparser.add_argument("--results_file",
            type = str
        )

    args = argparser.parse_args()
    main(args)
    