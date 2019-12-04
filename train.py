"""Train on synthetically generated data"""

import os
import pickle

from models import PlainRNN
from config import TRAINING_PARAMS, DATA_DIR

import trainer


if __name__ == "__main__":
    # train_lookup_path = os.path.join(DATA_DIR, 'lookups-tiny.pickle')
    # train_user_path = os.path.join(DATA_DIR, 'users-tiny.pickle')
    train_lookup_path = os.path.join(DATA_DIR, 'lookups-test.pickle')
    train_user_path = os.path.join(DATA_DIR, 'users-test.pickle')
    val_lookup_path = os.path.join(DATA_DIR, 'lookups-teeny.pickle')
    val_user_path = os.path.join(DATA_DIR, 'users-teeny.pickle')
    test_lookup_path = os.path.join(DATA_DIR, 'lookups-test.pickle')
    test_user_path = os.path.join(DATA_DIR, 'users-test.pickle')
    # test_lookup_path = os.path.join(DATA_DIR, 'lookups-teeny.pickle')
    # test_user_path = os.path.join(DATA_DIR, 'users-teeny.pickle')

    trainer.train_pipeline(PlainRNN, train_lookup_path, train_user_path, val_lookup_path, val_user_path, test_lookup_path, test_user_path, TRAINING_PARAMS)
