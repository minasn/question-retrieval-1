import sys
import argparse
import corpus

import numpy as np

from scipy import spatial
from evaluation import *
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim import Adam
import torch.autograd as autograd

def main(args):
    print args.model
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
    
    if args.model == 'lstm':
        print "training lstm"
        lstm = nn.LSTM(input_size=200, hidden_size=args.hidden_size)
        optimizer = Adam(lstm.parameters())
        if args.cuda:
            lstm.cuda()
    else:
        print "training cnn"
        cnn = nn.Conv1d(in_channels=200, out_channels=args.hidden_size, kernel_size = 3, padding = 1)
        optimizer = Adam(cnn.parameters())
        if args.cuda:
            cnn.cuda()

    # lstm tutorial: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    # lstm documentation: http://pytorch.org/docs/master/nn.html?highlight=nn%20lstm#torch.nn.LSTM
    # 
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
                    title_inputs = torch.cat(title_inputs).view(title_num_questions, 200, -1)
                else:
                    title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
                    title_inputs = torch.cat(title_inputs).view(title_num_questions, 200, -1)

            if args.model == 'lstm':
                title_out, title_hidden = lstm(title_inputs, title_hidden)
            else:
                title_out = cnn(title_inputs)

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
                    body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
                else:
                    body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
                    body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
            
            if args.model == 'lstm':
                body_out, body_hidden = lstm(body_inputs, body_hidden)
            else:
                body_out = cnn(body_inputs)

            average_body_out = average_questions(body_out, bodies, padding_id)
            count+=1

            # average body and title
            # representations of the questions as found by the LSTM
            hidden = (average_title_out + average_body_out) * 0.5
            print "train"
            print hidden.size()
            print hidden
            if args.cuda:
            	triples_vectors = hidden[torch.LongTensor(triples.ravel()).cuda()]
            else: 
            	triples_vectors = hidden[torch.LongTensor(triples.ravel())]
            print triples_vectors.size()
            triples_vectors = triples_vectors.view(triples.shape[0], triples.shape[1], args.hidden_size)
            print triples_vectors.size()

            query = triples_vectors[:, 0, :].unsqueeze(1)
            examples = triples_vectors[:, 1:, :]

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
                loss = F.multi_margin_loss(cos_similarity, targets, margin = 0.3).cuda()
            else:
                loss = F.multi_margin_loss(cos_similarity, targets, margin=0.3)
            total_loss += loss.cpu().data.numpy()[0]
            loss.backward()
            #print "average loss: " + str((total_loss/float(count)))

            optimizer.step() 

            evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, lstm)

def evaluation(args, padding_id, ids_corpus, vocab_map, embeddings, lstm):
    print "starting evaluation"
    val_data = corpus.read_annotations(args.test)
    val_batches = corpus.create_eval_batches(ids_corpus, val_data, padding_id)
    count = 0
    similarities = np.array([])
    titles, bodies, qlabels = val_batches[0]
    title_length, title_num_questions = titles.shape
    body_length, body_num_questions = bodies.shape
    title_embeddings, body_embeddings = corpus.get_embeddings(titles, bodies, vocab_map, embeddings)
    
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
            title_inputs = torch.cat(title_inputs).view(title_num_questions, 200, -1)
        else:
            title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
            title_inputs = torch.cat(title_inputs).view(title_num_questions, 200, -1)

    if args.model == 'lstm':
        title_out, title_hidden = lstm(title_inputs, title_hidden)
    else:
        title_out = cnn(title_inputs)

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
            body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
        else:
            body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
            body_inputs = torch.cat(body_inputs).view(body_num_questions, 200, -1)
    
    if args.model == 'lstm':
        body_out, body_hidden = lstm(body_inputs, body_hidden)
        #print body_out
    else:
        body_out = cnn(body_inputs)

    # average all words of each question from body_out
    average_body_out = average_questions(body_out, bodies, padding_id)

    # average body and title
    # representations of the questions as found by the LSTM
    # 560 x 100
    hidden = (average_title_out + average_body_out) * 0.5
    print "dev"
    print hidden.size()
    print hidden

    # if args.cuda:
    #     triples_vectors = hidden[torch.LongTensor(triples.ravel()).cuda()]
    # else: 
    #     triples_vectors = hidden[torch.LongTensor(triples.ravel())]
    # # triples_vectors = hidden[torch.LongTensor(triples.ravel())]
    # triples_vectors = triples_vectors.view(triples.shape[0], triples.shape[1], args.hidden_size)
    hidden = hidden.view(title_num_questions, 21, args.hidden_size)

    query = hidden[:, 0, :].unsqueeze(1)
    examples = hidden[:, 1:, :]

    cos_similarity = F.cosine_similarity(query, examples, dim=2)
    cos_similarity_np = cos_similarity.data.numpy()
    ranked_similarities = np.argsort(-1*cos_similarity_np, axis=1)
    positive_similarity = qlabels[ranked_similarities]
    # print positive_similarity

    evaluator = Evaluation(positive_similarity)
    print "precision at 1: " + str(evaluator.Precision(1))
    print "precision at 5: " + str(evaluator.Precision(5))
    print "MAP: " + str(evaluator.MAP())
    print "MRR: " + str(evaluator.MRR())

def cos_sim_func(triples_vectors):
    """Create an array of the cosine similarity scores of each vector
    in triple and the first vector in triple. Excludes the first vector.
    """
    cos_sim = np.vectorize(lambda x: 1 - spatial.distance.cosine(triples_vectors[0], x))
    # exclude the first question in triple
    return cos_sim(triples_vectors)[1:]

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
    
    args = argparser.parse_args()
    main(args)
    
