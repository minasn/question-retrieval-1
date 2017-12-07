import sys
import argparse
import corpus

import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd

def main(args):
    raw_corpus = corpus.read_corpus(args.corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    ids_corpus = corpus.map_corpus(vocab_map, raw_corpus)
    annotations = corpus.read_annotations(args.train)

    training_batches = corpus.create_batches(ids_corpus, annotations, args.batch_size, padding_id)
    
    lstm = nn.LSTM(input_size=200, hidden_size=args.hidden_size)

    # lstm tutorial: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    # lstm documentation: http://pytorch.org/docs/master/nn.html?highlight=nn%20lstm#torch.nn.LSTM
    # 

    for batch in training_batches:
        titles, bodies, triples = batch
        title_length, title_num_questions = titles.shape
        body_length, body_num_questions = bodies.shape
        title_embeddings, body_embeddings = corpus.get_embeddings(titles, bodies, vocab_map, embeddings)
        
        # title
        title_inputs = [autograd.Variable(torch.FloatTensor(title_embeddings))]
        title_inputs = torch.cat(title_inputs).view(title_length, title_num_questions, -1)

        # checking title_inputs dimensions
        # print len(title_inputs)
        # print len(title_inputs[0])
        # print len(title_inputs[0][0])

        title_hidden = (autograd.Variable(torch.zeros(1, title_num_questions, args.hidden_size)),
              autograd.Variable(torch.zeros((1, title_num_questions, args.hidden_size))))

        title_out, title_hidden = lstm(title_inputs, title_hidden)

        # print(len(title_out))
        # print(len(title_out[0]))
        # print(len(title_out[0][0]))

        # average all words of each question from title_out
        # title_out (max sequence length) x (batch size) x (hidden size)
        average_title_out = average_questions(title_out, titles, padding_id)
        # print "avg title out "
        # print len(average_title_out)
        # print "\n"

        # body
        body_inputs = [autograd.Variable(torch.FloatTensor(body_embeddings))]
        body_inputs = torch.cat(body_inputs).view(body_length, body_num_questions, -1)

        body_hidden = (autograd.Variable(torch.zeros(1, body_num_questions, args.hidden_size)),
              autograd.Variable(torch.zeros((1, body_num_questions, args.hidden_size))))
        
        body_out, body_hidden = lstm(body_inputs, body_hidden)

        # print(len(body_out))
        # print(len(body_out[0]))
        # print(len(body_out[0][0]))

        # average all words of each question from body_out
        # body_out (max sequence length) x (batch size) x (hidden size)
        average_body_out = average_questions(body_out, bodies, padding_id)
        # print "avg body out "
        # print len(average_body_out)
        # print "\n"

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
    