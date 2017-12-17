import sys
import os
import argparse

import numpy as np
import csv

import corpus
from evaluation import *
from meter import AUCMeter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam

def main(args):
    model = load_model(args)
    print "loaded " + args.model

    raw_corpus = corpus.read_corpus(args.corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    print("loaded embeddings")
    ids_corpus = corpus.map_corpus(vocab_map, raw_corpus)

    evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, model)

def load_model(args):   
    """Load either an LSTM or CNN.
    """ 
    if args.model == 'lstm':
        print("loading " + args.load_model)
        lstm = nn.LSTM(input_size=args.embedding_size, hidden_size=args.hidden_size)
        lstm.load_state_dict(torch.load(args.load_model))
        optimizer = Adam(lstm.parameters())
        if args.cuda:
            lstm.cuda()
        return lstm
    else:
        print("loading " + args.load_model)
        cnn = nn.Conv1d(in_channels=args.embedding_size, out_channels=args.hidden_size, kernel_size = 3, padding = 1)
        cnn.load_state_dict(torch.load(args.load_model))
        optimizer = Adam(cnn.parameters())
        if args.cuda:
            cnn.cuda()
        return cnn

def evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, model):
    """Calculate the AUC score of the model on Android data.
    """
    meter = AUCMeter()

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
        
        if args.model == 'lstm':
            if args.cuda:
                title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings).cuda())]
                title_inputs = torch.cat(title_inputs).view(title_length, title_num_questions, -1)

                title_hidden = (autograd.Variable(torch.zeros(1, title_num_questions, args.hidden_size).cuda()),
                      autograd.Variable(torch.zeros((1, title_num_questions, args.hidden_size)).cuda()))
            else:
                title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
                title_inputs = torch.cat(title_inputs).view(title_length, title_num_questions, -1)
                # title_inputs = torch.cat(title_inputs).view(title_num_questions, title_length, -1)

                title_hidden = (autograd.Variable(torch.zeros(1, title_num_questions, args.hidden_size)),
                      autograd.Variable(torch.zeros((1, title_num_questions, args.hidden_size))))
        else:
            if args.cuda:
                title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings).cuda())]
            else:
                title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
            title_inputs = torch.cat(title_inputs).transpose(0,1).transpose(1,2)

        if args.model == 'lstm':
            title_out, title_hidden = model(title_inputs, title_hidden)
        else:
            title_out = model(title_inputs)
            title_out = F.tanh(title_out)
            title_out = title_out.transpose(1,2).transpose(0,1)

        average_title_out = average_questions(title_out, titles, padding_id)

        # body
        if args.model == 'lstm':
            if args.cuda:
                body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings).cuda())]
                body_inputs = torch.cat(body_inputs).view(body_length, body_num_questions, -1)

                body_hidden = (autograd.Variable(torch.zeros(1, body_num_questions, args.hidden_size).cuda()),
                      autograd.Variable(torch.zeros((1, body_num_questions, args.hidden_size)).cuda()))
            else:
                body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
                body_inputs = torch.cat(body_inputs).view(body_length, body_num_questions, -1)

                body_hidden = (autograd.Variable(torch.zeros(1, body_num_questions, args.hidden_size)),
                      autograd.Variable(torch.zeros((1, body_num_questions, args.hidden_size))))
        else:
            if args.cuda:
                body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings).cuda())]
                #body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
            else:
                body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
                #body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
            body_inputs = torch.cat(body_inputs).transpose(0,1).transpose(1,2)
        
        if args.model == 'lstm':
            body_out, body_hidden = model(body_inputs, body_hidden)
        else:
            body_out = model(body_inputs)
            body_out = F.tanh(body_out)
            body_out = body_out.transpose(1,2).transpose(0,1)

        # average all words of each question from body_out
        average_body_out = average_questions(body_out, bodies, padding_id)

        # average body and title
        # representations of the questions as found by the CNN
        # 560 x 100
        hidden = (average_title_out + average_body_out) * 0.5

        query = torch.DoubleTensor(hidden[0].unsqueeze(0).cpu().data.numpy())
        examples = torch.DoubleTensor(hidden[1:].cpu().data.numpy())

        cos_similarity = F.cosine_similarity(query, examples, dim=1)
        qlabels = [float(qlabel) for qlabel in list(qlabels)]
        target = torch.DoubleTensor(qlabels)
        meter.add(cos_similarity, target)

    print meter.value(0.05)

def average_questions(hidden, ids, padding_id, eps=1e-10):
    """Average the outputs from the hidden states of questions, excluding padding.
    """
    # sequence (title or body) x questions x 1
    if args.cuda:
        mask = autograd.Variable(torch.from_numpy(1 * (ids != padding_id)).type(torch.FloatTensor).cuda().unsqueeze(2))
    else:
        mask = autograd.Variable(torch.from_numpy(1 * (ids != padding_id)).type(torch.FloatTensor).unsqueeze(2))
    # questions x hidden (=200)
    masked_sum = torch.sum(mask * hidden, dim=0)

    # questions x 1
    lengths = torch.sum(mask, dim=0)

    return masked_sum / (lengths + eps)

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
    argparser.add_argument("--model",
            type = str,
            default = "lstm"
        )
    argparser.add_argument("--embedding_size",
            type = int,
            default = 200
        )

    args = argparser.parse_args()
    main(args)
    