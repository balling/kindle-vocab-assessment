r"""Generic utilities."""

# everything loads from this file so do not make relative imports
import os
import shutil
import random
import pickle
import numpy as np

from collections import Counter
from collections import defaultdict, OrderedDict

import torch

# these are "special" tokens used often to handle language
# given a sentence like "A brown dog.", we add these tokens
# to make it:
#
#       <sos> A brown dog . <pad> ... <pad> <eos>
#
# where pad tokens are added to make ALL sentences in a
# minibatch to the same size!
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

SPECIAL_TOKENS = [PAD_TOKEN]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    # saves a copy of the model (+ properties) to filesystem
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))
