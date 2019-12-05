import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils

# from trainer.utils import NUM_LABELS, IX_TO_LABEL

PAD_IDX = 0
EMBED_SIZE = 303
HIDDEN_SIZE = 128
N_CLASS = 13


class PlainRNN(nn.Module):
    """
    Neural network responsible for ingesting a tokenized student 
    program, and spitting out a categorical prediction.

    We give you the following information:
        vocab_size: number of unique tokens 
        num_labels: number of output feedback labels
    """

    def __init__(self, weight, vocab_size, num_labels):
        super().__init__()
        # These modules define trainable parameters. Put things here like
        #   nn.Linear, nn.RNN, nn.Embedding
        self.embedding = nn.Embedding.from_pretrained( weight, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, bidirectional=True)
        self.projection = nn.Linear(2 * HIDDEN_SIZE, N_CLASS)
        self.word_projection = nn.Linear(2 * HIDDEN_SIZE, 1)

    def forward(self, token_seq, token_length):
        """
        Forward pass for your feedback prediction network.

        @param token_seq: batch_size x max_seq_length
            Example: torch.Tensor([[0,6,2,3],[0,2,5,3], ...])
            These define your PADDED programs after tokenization.

        @param token_length: batch_size
            Example: torch.Tensor([4,4, ...])
            These define your unpadded program lengths.

        This function should return the following:
        @param label_out: batch_size x num_labels
            Each index in this tensor represents the likelihood of predicting
            1. Unlike IRT, this is a multilabel prediction program so we need
            to have a likelihood for every feedback. NOTE: this is NOT categorical
            since we can have multiple feedback at once. 

            This will be given to F.binary_cross_entropy(...), just like IRT!
        """
        # embeddings = self.embedding(token_seq) # batch_size x max_seq_length x embed_size
        seq = token_seq.permute(0, 2, 1)
        # seen_seq, lookup_seq, level_seq, token_seq
        embeddings = self.embedding(seq[:, :, 3].long()).float()
        embeddings = torch.cat((embeddings, seq[:, :, 0:3],), 2)
        packed = rnn_utils.pack_padded_sequence(embeddings, token_length, batch_first=True, enforce_sorted=False)
        hiddens, (last_hidden, last_cell) = self.rnn(packed)
        hiddens = rnn_utils.pad_packed_sequence(hiddens, batch_first=True)[0]  # batch_size x max_seq_length x 2 hidden_size
        # batch_size x num_labels
        score = self.projection(torch.cat((last_hidden[0], last_hidden[1]), 1))
        word_scores = self.word_projection(hiddens).squeeze(2)
        # batch_size, embed_size, max_seq_length = token_seq.size()
        return torch.sigmoid(word_scores), score
