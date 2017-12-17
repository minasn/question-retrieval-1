import sys
import os
import argparse
import corpus

import numpy as np
import csv
from datetime import datetime
from meter import AUCMeter

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim import Adam
import torch.autograd as autograd

class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()

        self.args = args
        self.lin_layer_1 = nn.Linear(in_features = args.hidden_size, out_features = args.hidden_size)
        self.lin_layer_2 = nn.Linear(in_features = args.hidden_size, out_features = args.hidden_size)
        self.output_layer = nn.Linear(in_features = args.hidden_size, out_features = 2)

    def forward(self, inputs):
        hidden1 = self.lin_layer_1(inputs)
        activated1 = F.relu(hidden1)
        hidden2 = self.lin_layer_2(activated1)
        activated2 = F.relu(hidden2)
        output = self.output_layer(activated2)
        return output

def main(args):

    ubuntu_corpus = os.path.join(args.ubuntu_path, 'text_tokenized.txt.gz')
    android_corpus = os.path.join(args.android_path, 'corpus.tsv.gz')
    ubuntu_raw_corpus = corpus.read_corpus(ubuntu_corpus)
    android_raw_corpus = corpus.read_corpus(android_corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    print "loaded embeddings"

    ubuntu_ids_corpus = corpus.map_corpus(vocab_map, ubuntu_raw_corpus)
    android_ids_corpus = corpus.map_corpus(vocab_map, android_raw_corpus)
    ubuntu_train = os.path.join(args.ubuntu_path, 'train_random.txt')
    ubuntu_train_annotations = corpus.read_annotations(ubuntu_train)
    print len(ubuntu_train_annotations)
    ubuntu_training_batches = corpus.create_batches(ubuntu_ids_corpus, ubuntu_train_annotations, args.batch_size, padding_id)
    print "got ubuntu batches"

    if args.load_model:
        if args.model == 'lstm':
            print("loading " + args.load_model)
            lstm = nn.LSTM(input_size=300, hidden_size=args.hidden_size)
            lstm.load_state_dict(torch.load(args.load_model))
            optimizer = Adam(lstm.parameters())
            if args.cuda:
                lstm.cuda()
        else:
            print("loading " + args.load_model)
            cnn = nn.Conv1d(in_channels=300, out_channels=args.hidden_size, kernel_size = 3, padding = 1)
            cnn.load_state_dict(torch.load(args.load_model))
            optimizer = Adam(cnn.parameters())
            if args.cuda:
                cnn.cuda()
    else:
        if args.model == 'lstm':
            print "training lstm"
            lstm = nn.LSTM(input_size=300, hidden_size=args.hidden_size)
            optimizer = Adam(lstm.parameters())
            if args.cuda:
                lstm.cuda()
        else:
            print "training cnn"
            cnn = nn.Conv1d(in_channels=300, out_channels=args.hidden_size, kernel_size = 3, padding = 1)
            optimizer = Adam(cnn.parameters())
            if args.cuda:
                cnn.cuda()

    feed_forward = FeedForward(args)
    if args.cuda:
        feed_forward.cuda()
    feed_forward_optimizer = Adam(feed_forward.parameters(), lr=-0.001)

    android_dev_pos_path = os.path.join(args.android_path, 'dev.pos.txt')
    android_dev_neg_path = os.path.join(args.android_path, 'dev.neg.txt')
    android_dev_annotations = android_pairs_to_annotations(android_dev_pos_path, android_dev_neg_path)
    
    count = 1
    hidden_states = []
    total_loss = 0.0
    time_begin = datetime.now()
    for epoch in range(10):
        print "epoch = " + str(epoch)
        for batch in ubuntu_training_batches:

            titles, bodies, triples = batch

            optimizer.zero_grad()
            if count%10 == 0:
                print(count)
                print "average loss: " + str((total_loss/float(count)))
                print("time for 10 batches: " + str(datetime.now() - time_begin))
                time_begin = datetime.now()
            count += 1

            batch_size = len(batch[2])
            android_batch = corpus.domain_classifier_batch(android_ids_corpus, android_dev_annotations, batch_size, padding_id)
            android_titles, android_bodies, _ = android_batch

            if args.model == 'lstm':
                model = lstm
            else:
                model = cnn

            hidden_ubuntu = vectorize_question(args, batch, model, vocab_map, embeddings, padding_id)
            hidden_android = vectorize_question(args, android_batch, model, vocab_map, embeddings, padding_id)
            hidden_combined = torch.cat((hidden_ubuntu, hidden_android))
            input_size = int(hidden_combined.size()[0])

            output = feed_forward.forward(hidden_combined)

            domain_labels = [1]*int(hidden_ubuntu.size()[0]) + [0]*int(hidden_android.size()[0])
            domain_labels = autograd.Variable(torch.LongTensor(domain_labels))

            if args.cuda:
                triples_vectors = hidden_ubuntu[torch.LongTensor(triples.ravel()).cuda()]
            else: 
                triples_vectors = hidden_ubuntu[torch.LongTensor(triples.ravel())]
            triples_vectors = triples_vectors.view(triples.shape[0], triples.shape[1], args.hidden_size)

            query = triples_vectors[:, 0, :].unsqueeze(1)
            examples = triples_vectors[:, 1:, :]
            cos_similarity = F.cosine_similarity(query, examples, dim=2)
            if args.cuda:
                targets = autograd.Variable(torch.zeros(triples.shape[0]).type(torch.LongTensor).cuda())
            else:
                targets = autograd.Variable(torch.zeros(triples.shape[0]).type(torch.LongTensor))
            if args.cuda:
                encoder_loss = F.multi_margin_loss(cos_similarity, targets, margin = args.margin).cuda()
            else:
                encoder_loss = F.multi_margin_loss(cos_similarity, targets, margin=args.margin)
            total_loss += encoder_loss.cpu().data.numpy()[0]

            if args.cuda:
                domain_loss_func = nn.CrossEntropyLoss().cuda()
            else:
                domain_loss_func = nn.CrossEntropyLoss()
            domain_classifier_loss = domain_loss_func(output, domain_labels)
            combined_loss = encoder_loss - args.lam * domain_classifier_loss
            combined_loss.backward()

            optimizer.step()
            feed_forward_optimizer.step()

        evaluation(args, padding_id, android_ids_corpus, model, vocab_map, embeddings)

def evaluation(args, padding_id, android_ids_corpus, model, vocab_map, embeddings):
    print "starting evaluation"
    if args.model == 'lstm':
        lstm = model
    else:
        cnn = model

    meter = AUCMeter()

    android_test_pos_path = os.path.join(args.android_path, 'test.pos.txt')
    android_test_neg_path = os.path.join(args.android_path, 'test.neg.txt')
    android_test_annotations = android_pairs_to_annotations(android_test_pos_path, android_test_neg_path)
    android_test_batches = corpus.create_eval_batches(android_ids_corpus, android_test_annotations, padding_id)

    count = 0
    for batch in android_test_batches:
        titles, bodies, qlabels = batch

        if args.model == 'lstm':
            model = lstm
        else:
            model = cnn
        hidden = vectorize_question(args, batch, model, vocab_map, embeddings, padding_id)
        query = hidden[0].unsqueeze(0)
        examples = hidden[1:]
        cos_similarity = F.cosine_similarity(query, examples, dim=1)
        target = torch.DoubleTensor(qlabels)
        meter.add(cos_similarity.data, target)
    print meter.value(0.05) 

def android_pairs_to_annotations(pos_path, neg_path):
    android_positives = {}
    android_negatives = {}

    dev_pos_ids_X, dev_pos_Y = corpus.load_android_pairs(pos_path, True)
    for q1, q2 in dev_pos_ids_X:
        if q1 in android_positives:
            android_positives[q1].append(q2)
        else:
            android_positives[q1] = [q2]
        if q2 in android_positives:
            android_positives[q2].append(q1)
        else:
            android_positives[q2] = [q1]

    dev_neg_ids_X, dev_neg_Y = corpus.load_android_pairs(neg_path, False)
    for q1, q2 in dev_neg_ids_X:
        if q1 in android_negatives:
            android_negatives[q1].append(q2)
        else:
            android_negatives[q1] = [q2]
        if q2 in android_negatives:
            android_negatives[q2].append(q1)
        else:
            android_negatives[q2] = [q1]

    android_annotations = corpus.android_annotations(android_positives, android_negatives)
    return android_annotations

def vectorize_question(args, batch, model, vocab_map, embeddings, padding_id):

    if args.model == 'lstm':
        lstm = model
    else:
        cnn = model

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
    argparser.add_argument("--lam",
            type = str,
            default = 0.001
        )

    args = argparser.parse_args()
    main(args)
    
