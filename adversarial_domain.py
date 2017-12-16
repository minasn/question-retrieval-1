import sys
import os
import argparse
import corpus

import numpy as np
import csv
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim import Adam
import torch.autograd as autograd

def main(args):
    ubuntu_corpus = os.path.join(args.ubuntu_path, 'text_tokenized.txt.gz')
    android_corpus = os.path.join(args.android_path, 'corpus.tsv.gz')
    ubuntu_raw_corpus = corpus.read_corpus(ubuntu_corpus)
    android_raw_corpus = corpus.read_corpus(android_corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    ubuntu_ids_corpus = corpus.map_corpus(vocab_map, ubuntu_raw_corpus)
    android_ids_corpus = corpus.map_corpus(vocab_map, android_raw_corpus)
    ubuntu_train = os.path.join(args.ubuntu_path, 'train_random.txt')
    ubuntu_train_annotations = corpus.read_annotations(ubuntu_train)
    print len(ubuntu_train_annotations)
    ubuntu_training_batches = corpus.create_batches(ubuntu_ids_corpus, ubuntu_train_annotations, args.batch_size, padding_id)

    android_dev_positives = {}
    android_dev_negatives = {}
    android_dev_pos_path = os.path.join(args.android_path, 'dev.pos.txt')
    android_dev_neg_path = os.path.join(args.android_path, 'dev.neg.txt')
    android_test_pos_path = os.path.join(args.android_path, 'test.pos.txt')
    android_test_neg_path = os.path.join(args.android_path, 'test.neg.txt')

    dev_pos_ids_X, dev_pos_Y = corpus.load_android_pairs(android_dev_pos_path, True)
    for q1, q2 in dev_pos_ids_X:
        if q1 in android_dev_positives:
            android_dev_positives[q1].append(q2)
        else:
            android_dev_positives[q1] = [q2]
        if q2 in android_dev_positives:
            android_dev_positives[q2].append(q1)
        else:
            android_dev_positives[q2] = [q1]

    dev_neg_ids_X, dev_neg_Y = corpus.load_android_pairs(android_dev_neg_path, False)
    for q1, q2 in dev_neg_ids_X:
        if q1 in android_dev_negatives:
            android_dev_negatives[q1].append(q2)
        else:
            android_dev_negatives[q1] = [q2]
        if q2 in android_dev_negatives:
            android_dev_negatives[q2].append(q1)
        else:
            android_dev_negatives[q2] = [q1]

    android_dev_annotations = corpus.android_annotations(android_dev_positives, android_dev_negatives)
    print len(android_dev_annotations)
    
    print(1/0)
    count = 1
    hidden_states = []
    total_loss = 0.0
    time_begin = datetime.now()
    for epoch in range(10):
        print "epoch = " + str(epoch)
        for batch in ubuntu_training_batches:
            optimizer.zero_grad()
            if count%10 == 0:
                print(count)
                print "average loss: " + str((total_loss/float(count)))
                print("time for 10 batches: " + str(datetime.now() - time_begin))
                time_begin = datetime.now()
            count += 1

            hidden_ubuntu = vectorize_question(args, batch)

            if args.cuda:
                triples_vectors = hidden[torch.LongTensor(triples.ravel()).cuda()]
            else: 
                triples_vectors = hidden[torch.LongTensor(triples.ravel())]
            triples_vectors = triples_vectors.view(triples.shape[0], triples.shape[1], args.hidden_size)

            query = triples_vectors[:, 0, :].unsqueeze(1)
            examples = triples_vectors[:, 1:, :]
            cos_similarity = F.cosine_similarity(query, examples, dim=2)
            if args.cuda:
                targets = autograd.Variable(torch.zeros(triples.shape[0]).type(torch.LongTensor).cuda())
            else:
                targets = autograd.Variable(torch.zeros(triples.shape[0]).type(torch.LongTensor))
            if args.cuda:
                loss = F.multi_margin_loss(cos_similarity, targets, margin = args.margin).cuda()
            else:
                loss = F.multi_margin_loss(cos_similarity, targets, margin=args.margin)
            total_loss += loss.cpu().data.numpy()[0]
            loss.backward()
            optimizer.step() 

def vectorize_question(args, batch):
    titles, bodies, triples = batch
    title_length, title_num_questions = titles.shape
    body_length, body_num_questions = bodies.shape
    title_embeddings, body_embeddings = corpus.get_embeddings(titles, bodies, vocab_map, embeddings)
    
    # title
    if args.model == 'lstm':
        if args.cuda:
            title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings).cuda())]
            title_inputs = torch.cat(title_inputs).view(title_length, title_num_questions, -1)
            # title_inputs = torch.cat(title_inputs).view(title_num_questions, title_length, -1)

            title_hidden = (autograd.Variable(torch.zeros(1, title_num_questions, args.hidden_size).cuda()),
                  autograd.Variable(torch.zeros((1, title_num_questions, args.hidden_size)).cuda()))
            # title_hidden = (autograd.Variable(torch.zeros(1, title_length, args.hidden_size)),
            #       autograd.Variable(torch.zeros((1, title_length, args.hidden_size))))
        else:
            title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
            title_inputs = torch.cat(title_inputs).view(title_length, title_num_questions, -1)
            # title_inputs = torch.cat(title_inputs).view(title_num_questions, title_length, -1)

            title_hidden = (autograd.Variable(torch.zeros(1, title_num_questions, args.hidden_size)),
                  autograd.Variable(torch.zeros((1, title_num_questions, args.hidden_size))))
    else:
        if args.cuda:
            title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings).cuda())]
            #title_inputs = torch.cat(title_inputs).view(title_num_questions, 200, -1)
        else:
            title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
        title_inputs = torch.cat(title_inputs).transpose(0,1).transpose(1,2)

    if args.model == 'lstm':
        title_out, title_hidden = lstm(title_inputs, title_hidden)
    else:
        title_out = cnn(title_inputs)
        title_out = F.tanh(title_out)
        title_out = title_out.transpose(1,2).transpose(0,1)
        
    # average all words of each question from title_out
    # title_out (max sequence length) x (batch size) x (hidden size)
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
        else:
            body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
        body_inputs = torch.cat(body_inputs).transpose(0,1).transpose(1,2)
    
    if args.model == 'lstm':
        body_out, body_hidden = lstm(body_inputs, body_hidden)
    else:
        body_out = cnn(body_inputs)
        body_out = F.tanh(body_out)
        body_out = body_out.transpose(1,2).transpose(0,1)

    average_body_out = average_questions(body_out, bodies, padding_id)

    # average body and title
    # representations of the questions as found by the LSTM
    hidden = (average_title_out + average_body_out) * 0.5

    return hidden


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ubuntu_path",
            type = str
        )
    argparser.add_argument("--android_path",
            type = str
        )
    argparser.add_argument("--embeddings",
            type = str,
            default = ""
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 25
        )
    argparser.add_argument("--hidden_size",
            type = int,
            default = 100
        )
    argparser.add_argument("--model",
            type = str,
            default = "lstm"
        )
    argparser.add_argument("--cuda",
            type = int,
            default = 0
        )
    argparser.add_argument("--load_model",
            type = str,
            default = ""
        )
    argparser.add_argument("--save_model", 
            type = int,
            default = 1
        )
    argparser.add_argument("--results_file",
            type = str
        )
    argparser.add_argument("--margin",
            type = str,
            default = 0.3
        )

    args = argparser.parse_args()
    main(args)
    
