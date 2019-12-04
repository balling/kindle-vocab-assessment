import os
import sys
import random
import pickle
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import zip_longest
from torch.utils.data.dataset import Dataset

from .utils import (
    PAD_TOKEN,
    SPECIAL_TOKENS,
)


def chunks(lst, n, padvalue=None):
    """
    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    Yield successive n-sized chunks from lst.
    """
    return zip_longest(*[iter(lst)]*n, fillvalue=padvalue)


class LookupDataset(Dataset):
    def __init__(self, lookup_path, user_path, vocab_path, levels_path, max_seq_len=50):
        super(LookupDataset, self).__init__()

        self.lookup_path = lookup_path
        self.user_path = user_path
        self.vocab_path = vocab_path
        self.levels_path = levels_path
        self.max_seq_len = max_seq_len

        self.vocab = self.create_vocab(vocab_path)
        self.w2i, self.i2w, self.words = self.vocab['w2i'], self.vocab['i2w'], self.vocab['words']
        self.levels = self.create_levels(levels_path)
        self.vocab_size = len(self.w2i)

        user_seqs, token_seqs, level_seqs, lookup_seqs, seen_seqs, token_lens = self.process_lookup(
            lookup_path)

        self.user_seqs = user_seqs
        self.token_seqs = token_seqs
        self.lookup_seqs = lookup_seqs
        self.level_seqs = level_seqs
        self.seen_seqs = seen_seqs
        self.token_lens = token_lens

        word_labels, ability_labels = self.process_labels(user_path)
        self.word_labels = word_labels
        self.ability_labels = ability_labels

    def create_levels(self, path):
        with open(path) as f:
            word_levels = json.load(f)
        levels = [word_levels[self.i2w[i]] for i in self.words]
        return levels
    
    def create_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as fp:
            data = pickle.load(fp)
        vocab, words = data['vocab'], data['words']
        w2i, i2w = dict(), dict()
        for i in range(len(SPECIAL_TOKENS)):
            i2w[i] = SPECIAL_TOKENS[i]
            w2i[SPECIAL_TOKENS[i]] = i
        for j in range(len(vocab)):
            i = j + len(SPECIAL_TOKENS)
            i2w[i] = vocab[j]
            w2i[vocab[j]] = i
        assert len(w2i) == len(i2w)
        print("Vocabulary of %i keys created." % len(w2i))
        return dict(w2i=w2i, i2w=i2w, words=np.vectorize(w2i.get)(words))

    def process_lookup(self, lookup_path):
        user_seqs, token_seqs, level_seqs, lookup_seqs, seen_seqs, token_lens = [], [], [], [], [], []
        with open(lookup_path, 'rb') as fp:
            lookups = pickle.load(fp)
        for user_id in lookups:
            lookup, seen = lookups[user_id]['lookup'], lookups[user_id]['seen']
            seqs, remain = divmod(len(lookup), self.max_seq_len)
            for i, chunk in enumerate(chunks(list(zip(lookup, seen, self.words, self.levels)), self.max_seq_len, (0, 0, self.w2i[PAD_TOKEN], 0))):
                user_seqs.append(user_id)
                token_lens.append(self.max_seq_len if i < seqs else remain)
                lookup_chunk, seen_chunk, word_chunk, level_chunk = zip(*list(chunk))
                token_seqs.append(np.array(word_chunk))
                lookup_seqs.append(np.array(lookup_chunk))
                seen_seqs.append(np.array(seen_chunk))
                level_seqs.append(np.array(level_chunk))
        print('Processed {} chunks of lookup sequence'.format(len(user_seqs)))
        return user_seqs, token_seqs, level_seqs, lookup_seqs, seen_seqs, token_lens

    def __len__(self):
        return len(self.token_seqs)

    def process_labels(self, user_path):
        abilities = []
        words = []
        with open(user_path, 'rb') as fp:
            users = pickle.load(fp)
        for user_id, tokens in zip(self.user_seqs, self.token_seqs):
            user = users[user_id]
            abilities.append(user['ability'])
            test = [int(user['known'][int(tok)-len(SPECIAL_TOKENS)])
                    for tok in tokens]
            words.append(np.array(test))
        print('Processed {} labels'.format(len(words)))
        return self.to_torch(words), self.to_torch(abilities)

    def to_torch(self, l):
        l = np.array(l).astype(np.int)
        return torch.from_numpy(l).float()

    def __getitem__(self, index):
        token_seq, lookup_seq, seen_seq, token_len, level_seq = self.token_seqs[index], self.lookup_seqs[index], self.seen_seqs[index], self.token_lens[index], self.level_seqs[index]
        token_seq = torch.from_numpy(np.array([seen_seq, lookup_seq, level_seq, token_seq])).float()
        word_label = self.word_labels[index]
        ability_label = self.ability_labels[index]
        return token_seq, token_len, word_label, ability_label
