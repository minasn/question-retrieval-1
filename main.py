import sys
import os
import argparse
import corpus

import numpy as np
import csv

from evaluation import *
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torch.autograd as autograd

def main(args):
    time1 = datetime.now()
    raw_corpus = corpus.read_corpus(args.corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    print("loaded embeddings")
    ids_corpus = corpus.map_corpus(vocab_map, raw_corpus)
    annotations = corpus.read_annotations(args.train)
    print("got annotations")

    training_batches = corpus.create_batches(ids_corpus, annotations, args.batch_size, padding_id)
    print("got batches")

    time2 = datetime.now()
    print "time to preprocess: " + str(time2-time1)

    if args.model == 'cnn':
        args.margin = 0.2
    
    if args.load_model:
        if args.model == 'lstm':
            print("loading " + args.load_model)
            lstm = nn.LSTM(input_size=args.embedding_size, hidden_size=args.hidden_size)
            lstm.load_state_dict(torch.load(args.load_model))
            optimizer = Adam(lstm.parameters())
            if args.cuda:
                lstm.cuda()
        else:
            print("loading " + args.load_model)
            cnn = nn.Conv1d(in_channels=args.embedding_size, out_channels=args.hidden_size, kernel_size = 3, padding = 1)
            cnn.load_state_dict(torch.load(args.load_model))
            optimizer = Adam(cnn.parameters())
            if args.cuda:
                cnn.cuda()
    else:
        if args.model == 'lstm':
            print "training lstm"
            lstm = nn.LSTM(input_size=args.embedding_size, hidden_size=args.hidden_size)
            optimizer = Adam(lstm.parameters())
            if args.cuda:
                lstm.cuda()
        else:
            print "training cnn"
            cnn = nn.Conv1d(in_channels=args.embedding_size, out_channels=args.hidden_size, kernel_size = 3, padding = 1)
            optimizer = Adam(cnn.parameters())
            if args.cuda:
                cnn.cuda()

    if args.save_model:
        if args.model == 'lstm':
            lstm_model_nums = []
            for d in os.listdir("lstm_models"):
                if "lstm_model" in d:
                    num = int(d[len("lstm_models")-1:])
                    lstm_model_nums.append(num)
            if len(lstm_model_nums) > 0:
                new_model_num = max(lstm_model_nums) + 1
            else:
                new_model_num = 0
            print("creating new model " + "lstm_models/lstm_model" + str(new_model_num))
            os.makedirs("lstm_models/lstm_model" + str(new_model_num))
        else:
            cnn_model_nums = []
            for d in os.listdir("cnn_models"):
                if "cnn_model" in d:
                    num = int(d[len("cnn_models")-1:])
                    cnn_model_nums.append(num)
            if len(cnn_model_nums) > 0:
                new_model_num = max(cnn_model_nums) + 1
            else:
                new_model_num = 0
            print("creating new model " + "cnn_models/cnn_model" + str(new_model_num))
            os.makedirs("cnn_models/cnn_model" + str(new_model_num))


    # lstm tutorial: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    # lstm documentation: http://pytorch.org/docs/master/nn.html?highlight=nn%20lstm#torch.nn.LSTM
    
    count = 1
    hidden_states = []
    total_loss = 0.0
    time_begin = datetime.now()
    for epoch in range(10):
        print "epoch = " + str(epoch)
        for batch in training_batches:
            optimizer.zero_grad()
            if count%10 == 0:
                print(count)
                print "average loss: " + str((total_loss/float(count)))
                print("time for 10 batches: " + str(datetime.now() - time_begin))
                time_begin = datetime.now()
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
                    # body_inputs = torch.cat(body_inputs).view(body_num_questions, body_length, -1)

                    body_hidden = (autograd.Variable(torch.zeros(1, body_num_questions, args.hidden_size).cuda()),
                          autograd.Variable(torch.zeros((1, body_num_questions, args.hidden_size)).cuda()))
                    # body_hidden = (autograd.Variable(torch.zeros(1, body_length, args.hidden_size)),
                    #       autograd.Variable(torch.zeros((1, body_length, args.hidden_size))))
                else:
                    body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
                    body_inputs = torch.cat(body_inputs).view(body_length, body_num_questions, -1)
                    # body_inputs = torch.cat(body_inputs).view(body_num_questions, body_length, -1)

                    body_hidden = (autograd.Variable(torch.zeros(1, body_num_questions, args.hidden_size)),
                          autograd.Variable(torch.zeros((1, body_num_questions, args.hidden_size))))
                    # body_hidden = (autograd.Variable(torch.zeros(1, body_length, args.hidden_size)),
                    #       aut
            else:
                if args.cuda:
                    body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings).cuda())]
                    #body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
                else:
                    body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
                    #body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
                body_inputs = torch.cat(body_inputs).transpose(0,1).transpose(1,2)
            
            if args.model == 'lstm':
                body_out, body_hidden = lstm(body_inputs, body_hidden)
            else:
                body_out = cnn(body_inputs)
                body_out = F.tanh(body_out)
                body_out = body_out.transpose(1,2).transpose(0,1)
                #body_out = body_out.view(body_length, body_num_questions, -1)

            average_body_out = average_questions(body_out, bodies, padding_id)
            count+=1

            # average body and title
            # representations of the questions as found by the LSTM
            hidden = (average_title_out + average_body_out) * 0.5
            # print "train"
            # print hidden.size()
            # print hidden
            if args.cuda:
                triples_vectors = hidden[torch.LongTensor(triples.ravel()).cuda()]
            else: 
            	triples_vectors = hidden[torch.LongTensor(triples.ravel())]
            # print triples_vectors.size()
            triples_vectors = triples_vectors.view(triples.shape[0], triples.shape[1], args.hidden_size)
            # print triples_vectors.size()

            query = triples_vectors[:, 0, :].unsqueeze(1)
            examples = triples_vectors[:, 1:, :]

            # print query.size()
            # print query
            # print examples.size()
            # print examples

            cos_similarity = F.cosine_similarity(query, examples, dim=2)
            # print "training"
            # print cos_similarity.size()
            if args.cuda:
                targets = autograd.Variable(torch.zeros(triples.shape[0]).type(torch.LongTensor).cuda())
            else:
                targets = autograd.Variable(torch.zeros(triples.shape[0]).type(torch.LongTensor))
            # outputs a Variable
            # By default, the losses are averaged over observations for each minibatch
            if args.cuda:
                loss = F.multi_margin_loss(cos_similarity, targets, margin = args.margin).cuda()
            else:
                loss = F.multi_margin_loss(cos_similarity, targets, margin=args.margin)
            total_loss += loss.cpu().data.numpy()[0]
            loss.backward()
            #print "average loss: " + str((total_loss/float(count)))

            optimizer.step() 

        result_headers = ['Epoch', 'MAP', 'MRR', 'P@1', 'P@5']
        with open(os.path.join(sys.path[0], args.results_file), 'a') as evaluate_file:
            writer = csv.writer(evaluate_file, dialect='excel')
            writer.writerow(result_headers)

        if args.model == 'lstm':
            evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, lstm, epoch)
        else:
            evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, cnn, epoch)

        if args.save_model:
            # saving the model
            if args.model == 'lstm':
                print "Saving lstm model epoch " + str(epoch) + " to lstm_model" + str(new_model_num)
                torch.save(lstm.state_dict(), "lstm_models/lstm_model" + str(new_model_num) + '/' + "epoch" + str(epoch))
            else:
                print "Saving cnn model epoch " + str(epoch) + " to cnn_model" + str(new_model_num)
                torch.save(cnn.state_dict(), "cnn_models/cnn_model" + str(new_model_num) + '/' + "epoch" + str(epoch))

def evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, model, epoch):
    print "starting evaluation"
    val_data = corpus.read_annotations(args.test)
    print "number of lines in test data: " + str(len(val_data))
    val_batches = corpus.create_eval_batches(ids_corpus, val_data, padding_id)
    count = 0
    similarities = []

    if args.model == 'lstm':
        lstm = model
    else:
        cnn = model

    for batch in val_batches:
        titles, bodies, qlabels = batch
        # print "Titles"
        # print titles.shape
        # print titles
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
                #title_inputs = torch.cat(title_inputs).view(title_num_questions, 200, -1)
            else:
                title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
                #title_inputs = torch.cat(title_inputs).view(title_num_questions, 200, -1)
            title_inputs = torch.cat(title_inputs).transpose(0,1).transpose(1,2)

        if args.model == 'lstm':
            title_out, title_hidden = lstm(title_inputs, title_hidden)
        else:
            title_out = cnn(title_inputs)
            title_out = F.tanh(title_out)
            title_out = title_out.transpose(1,2).transpose(0,1)
            #title_out = title_out.view(title_length, title_num_questions, -1)

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
            body_out, body_hidden = lstm(body_inputs, body_hidden)
        else:
            body_out = cnn(body_inputs)
            body_out = F.tanh(body_out)
            body_out = body_out.transpose(1,2).transpose(0,1)
            #body_out = body_out.view(body_length, body_num_questions, -1)

        # average all words of each question from body_out
        average_body_out = average_questions(body_out, bodies, padding_id)

        # average body and title
        # representations of the questions as found by the LSTM
        # 560 x 100
        hidden = (average_title_out + average_body_out) * 0.5
        # print "dev"
        # print hidden.size()
        # print hidden

        query = hidden[0].unsqueeze(0)
        examples = hidden[1:]

        # print query.size()
        # print query
        # print examples.size()
        # print examples

        cos_similarity = F.cosine_similarity(query, examples, dim=1)
        cos_similarity_np = cos_similarity.cpu().data.numpy()
        # print cos_similarity_np
        # print cos_similarity_np.shape
        ranked_similarities = np.argsort(-1*cos_similarity_np)
        positive_similarity = qlabels[ranked_similarities]
        similarities.append(positive_similarity)

    evaluator = Evaluation(similarities)
    metrics = [epoch, evaluator.MAP(), evaluator.MRR(), str(evaluator.Precision(1)), str(evaluator.Precision(5))]
    print "precision at 1: " + str(evaluator.Precision(1))
    print "precision at 5: " + str(evaluator.Precision(5))
    print "MAP: " + str(evaluator.MAP())
    print "MRR: " + str(evaluator.MRR())

    with open(os.path.join(sys.path[0],args.results_file), 'a') as evaluate_file:
        writer = csv.writer(evaluate_file, dialect='excel')
        writer.writerow(metrics)

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
    argparser.add_argument("--train",
            type = str,
            default = ""
        )
    argparser.add_argument("--test",
            type = str,
            default = ""
        )
    argparser.add_argument("--dev",
            type = str,
            default = ""
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
    argparser.add_argument("--embedding_size",
            type = int,
            default = 200
        )

    args = argparser.parse_args()
    main(args)
    