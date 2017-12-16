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
    ubuntu_training_batches = corpus.create_batches(unbuntu_ids_corpus, unbuntu_train_annotations, args.batch_size, padding_id)

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
    
