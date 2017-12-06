import sys
import argparse
import corpus

import torch.nn as nn
import torch.autograd as autograd

def main(args):
    raw_corpus = corpus.read_corpus(args.corpus)
    list_words, vocab_map, embeddings, padding_id = corpus.load_embeddings(corpus.load_embedding_iterator(args.embeddings))
    ids_corpus = corpus.map_corpus(vocab_map, raw_corpus)
    annotations = corpus.read_annotations(args.train)

    training_batches = corpus.create_batches(ids_corpus, annotations, args.batch_size, padding_id)
    
    lstm = nn.LSTM(input_size=200, hidden_size=200)

    for batch in training_batches:
        titles, bodies, triples = batch
        title_embeddings, body_embeddings = corpus.get_embeddings(titles, bodies, vocab_map, embeddings)
        
        # title
        title_inputs = [autograd.Variable(title_embeddings)]
        title_inputs = torch.cat(title_inputs).view(len(title_inputs), 1, -1)

        title_hidden = (autograd.Variable(torch.zeros(1, 1, 3)),
              autograd.Variable(torch.zeros((1, 1, 3))))

        out, title_hidden = lstm(title_inputs, title_hidden)

        print out
        print out.shape

        # body
        body_inputs = [autograd.Variable(body_embeddings)]
        body_inputs = torch.cat(body_inputs).view(len(body_inputs), 1, -1)

        body_hidden = (autograd.Variable(torch.zeros(1, 1, 3)),
              autograd.Variable(torch.zeros((1, 1, 3))))
        
        out, body_hidden = lstm(body_inputs, body_hidden)

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
    
    args = argparser.parse_args()
    main(args)

    
