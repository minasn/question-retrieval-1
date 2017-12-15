import sys
import argparse
import corpus

import numpy as np

from scipy import spatial

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim import Adam
import torch.autograd as autograd

def main(args):
    raw_corpus = corpus.read_corpus(args.corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    print("loaded embeddings")
    ids_corpus = corpus.map_corpus(vocab_map, raw_corpus)
    annotations = corpus.read_annotations(args.train)
    print("got annotations")

    training_batches = corpus.create_batches(ids_corpus, annotations, args.batch_size, padding_id)
    print("got batches")
    
    lstm = nn.LSTM(input_size=200, hidden_size=args.hidden_size)
    optimizer = Adam(lstm.parameters())

    # lstm tutorial: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    # lstm documentation: http://pytorch.org/docs/master/nn.html?highlight=nn%20lstm#torch.nn.LSTM
    # 
    count = 1
    hidden_states = []
    total_loss = 0.0
    for batch in training_batches:
        optimizer.zero_grad()
        if count%10 == 0:
            print(count)
        titles, bodies, triples = batch
        title_length, title_num_questions = titles.shape
        # print "title_length: " + str(title_length)
        # print "title_num_questions: " + str(title_num_questions)
        body_length, body_num_questions = bodies.shape
        # print "title length: " + str(title_length)
        # print "title num questions: " + str(title_num_questions)
        # print "body length: " + str(body_length)
        # print "body_num_questions: " + str(body_num_questions)
        title_embeddings, body_embeddings = corpus.get_embeddings(titles, bodies, vocab_map, embeddings)
        
        # title
        title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
        title_inputs = torch.cat(title_inputs).view(title_length, title_num_questions, -1)
        # title_inputs = torch.cat(title_inputs).view(title_num_questions, title_length, -1)

<<<<<<< HEAD
        # checking title_inputs dimensions
        # print "title_inputs dimensions"
=======
        # print "checking title_inputs dimensions"
>>>>>>> a26a2d2086666a63811a2aa910caeaa8e1874f75
        # print len(title_inputs)
        # print len(title_inputs[0])
        # print len(title_inputs[0][0])

        title_hidden = (autograd.Variable(torch.zeros(1, title_num_questions, args.hidden_size)),
              autograd.Variable(torch.zeros((1, title_num_questions, args.hidden_size))))
        # title_hidden = (autograd.Variable(torch.zeros(1, title_length, args.hidden_size)),
        #       autograd.Variable(torch.zeros((1, title_length, args.hidden_size))))

        title_out, title_hidden = lstm(title_inputs, title_hidden)

<<<<<<< HEAD
        # print "title_out dimensions"
=======
        # print title_out.size()
        # print title_out

>>>>>>> a26a2d2086666a63811a2aa910caeaa8e1874f75
        # print(len(title_out))
        # print(len(title_out[0]))
        # print(len(title_out[0][0]))

        # average all words of each question from title_out
        # title_out (max sequence length) x (batch size) x (hidden size)
        average_title_out = average_questions(title_out, titles, padding_id)
        # print "avg title out "
        # print type(average_title_out)
        # print len(average_title_out)
        # print average_title_out.size()
        # print "\n"

        # body
        body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
        body_inputs = torch.cat(body_inputs).view(body_length, body_num_questions, -1)
        # body_inputs = torch.cat(body_inputs).view(body_num_questions, body_length, -1)


        # checking body_inputs dimensions
        # print "body_inputs dimensions"
        # print len(body_inputs)
        # print len(body_inputs[0])
        # print len(body_inputs[0][0])

        body_hidden = (autograd.Variable(torch.zeros(1, body_num_questions, args.hidden_size)),
              autograd.Variable(torch.zeros((1, body_num_questions, args.hidden_size))))
        # body_hidden = (autograd.Variable(torch.zeros(1, body_length, args.hidden_size)),
        #       autograd.Variable(torch.zeros((1, body_length, args.hidden_size))))
        
        body_out, body_hidden = lstm(body_inputs, body_hidden)

        # print "body out"
        # print(len(body_out))
        # print(len(body_out[0]))
        # print(len(body_out[0][0]))

        # average all words of each question from body_out
        # body_out (max sequence length) x (batch size) x (hidden size)
        average_body_out = average_questions(body_out, bodies, padding_id)
        #hidden_states.append(average_body_out)
        # print triples
        # print average_title_out
        # print average_body_out
        # print "avg body out "
        # print len(average_body_out)
        # print "\n"
        count+=1

        # average body and title
        # representations of the questions as found by the LSTM
        # 560 x 100
        hidden = (average_title_out + average_body_out) * 0.5
        print hidden
        print type(hidden)
        print hidden.size

        # triples_vectors is a matrix of the vectors representing the questions
        # as indicated by the indices in triples 
        # triples_vectors = np.vectorize(lambda x: hidden[x])(triples)
        # print "num things in batch: " + str(len(triples))
        # print triples_vectors.shape
        # print triples_vectors

        # print hidden
        # print triples.ravel()
        # print triples.ravel().shape

        triples_vectors = hidden[torch.LongTensor(triples.ravel())]
        triples_vectors = triples_vectors.view(triples.shape[0], triples.shape[1], args.hidden_size)

        # print triples_vectors

        query = triples_vectors[:, 0, :].unsqueeze(1)
        examples = triples_vectors[:, 1:, :]

        cos_similarity = F.cosine_similarity(query, examples, dim=2)

        # input matrix to the loss funcion of dimensions (questions x 21)
        # questions is the batch size
        # s(0, 1), s(0, 2), ..., s(0, 21)
        # inputs = np.apply_along_axis(cos_sim_func, 0, triples_vectors)
        # print "inputs"
        # print inputs.shape
        # # does this need be (21, 1)? to be a column of 0's or just (21)
        targets = autograd.Variable(torch.zeros(triples.shape[0]).type(torch.LongTensor))

        # outputs a Variable
        # By default, the losses are averaged over observations for each minibatch.
        loss = F.multi_margin_loss(cos_similarity, targets)
        total_loss += loss.cpu().data.numpy()[0]
        # print "did loss function"
        # loss = loss_function(inputs, targets)
        loss.backward()
        print "average loss: " + str((total_loss/float(count)))

        optimizer.step() 

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
    
    args = argparser.parse_args()
    main(args)
    