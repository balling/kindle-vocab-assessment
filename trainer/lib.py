import os
import sys
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from .utils import AverageMeter, save_checkpoint
from .datasets import LookupDataset


def train_pipeline(model_class, train_lookup_path, train_user_path, val_lookup_path, val_user_path, test_lookup_path, test_user_path, config):
    device = torch.device('cpu')  # no CUDA support for now

    # reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    if not os.path.isdir(config['out_dir']):
        os.makedirs(config['out_dir'])

    # load the dataset! this might be new for you guys but usually, we wrap
    # data into Dataset classes. 
    train_dataset = LookupDataset(train_lookup_path, train_user_path, vocab_path=config['vocab_path'], max_seq_len=config['max_seq_len'])
    val_dataset = LookupDataset(val_lookup_path, val_user_path, vocab_path=config['vocab_path'], max_seq_len=config['max_seq_len'])
    test_dataset = LookupDataset(test_lookup_path, test_user_path, vocab_path=config['vocab_path'], max_seq_len=config['max_seq_len'])

    # We use a Loader that wraps around a Dataset class to return minibatches...
    # "shuffle" means we randomly pick rows from the full set. We only do this in training
    # because it helps us not memorize the order of inputs.
    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # this instantiates our model
    model = model_class(vocab_size=2, num_labels=3)
    model = model.to(device)
    # initialize our optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()  # utility for tracking loss / accuracy
        word_acc_meter = AverageMeter()
        ability_acc_meter = AverageMeter()

        word_label_arr, ability_label_arr, word_pred_arr, ability_pred_arr = [], [], [], []
        pbar = tqdm(len(train_loader))

        for batch_idx, (token_seq, token_len, word_label, ability_label) in enumerate(train_loader):
            batch_size = len(token_seq)
            token_seq = token_seq.to(device)  # sequence of token indexes
            token_len = token_len.to(device)  # length of a non-padded sentence
            word_label = word_label.to(device)
            ability_label = ability_label.to(device)

            optimizer.zero_grad()
            word_label_out, ability_label_out = model(token_seq, token_len)  # label_out is the prediction
            loss = F.binary_cross_entropy(word_label_out, word_label) + F.binary_cross_entropy(ability_label_out, ability_label) # same loss as in IRT!

            # loss.backward()
            loss_meter.update(loss.item(), batch_size)

            word_pred_npy = torch.round(word_label_out).detach().numpy()
            word_label_npy = word_label.detach().numpy()
            ability_pred_npy = torch.round(ability_label_out).detach().numpy()
            ability_label_npy = ability_label.detach().numpy()

            optimizer.step()
            word_acc = np.mean(word_pred_npy == word_label_npy)
            word_acc_meter.update(word_acc, batch_size)
            word_label_arr.append(word_label_npy)
            word_pred_arr.append(word_pred_npy)
            ability_acc = np.mean(ability_pred_npy == ability_label_npy)
            ability_acc_meter.update(ability_acc, batch_size)
            ability_label_arr.append(ability_label_npy)
            ability_pred_arr.append(ability_pred_npy)

            pbar.set_postfix({'Loss': loss_meter.avg, 'Word Accuracy': word_acc_meter.avg, 'Ability Accuracy': ability_acc_meter.avg})
            pbar.update()

        pbar.close()

        word_label_arr = np.concatenate(word_label_arr).flatten()
        word_pred_arr = np.concatenate(word_pred_arr).flatten()
        ability_label_arr = np.concatenate(ability_label_arr).flatten()
        ability_pred_arr = np.concatenate(ability_pred_arr).flatten()

        word_acc = np.mean(word_pred_arr == word_label_arr)
        ability_acc = np.mean(ability_pred_arr == ability_label_arr)
        word_f1 = f1_score(word_label_arr, word_pred_arr)
        ability_f1 = f1_score(ability_label_arr, ability_pred_arr, average='micro')

        print('====> Epoch: {}\tLoss: {:.4f}\tWord Accuracy: {:.4f}\tWord F1: {:.4f}\tAbility Accuracy: {:.4f}\tAbility F1: {:.4f}'.format(
            epoch, loss_meter.avg, word_acc, word_f1, ability_acc, ability_f1))
        
        return loss_meter.avg, word_acc, word_f1, ability_acc, ability_f1


    def test(epoch, loader, name='Test'):
        model.eval()
        loss_meter = AverageMeter()
        word_label_arr, ability_label_arr, word_pred_arr, ability_pred_arr = [], [], [], []

        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for (token_seq, token_len, word_label, ability_label) in loader:
                    assert word_label is not None
                    assert ability_label is not None
                    batch_size = len(token_seq)
                    token_seq = token_seq.to(device)  # sequence of token indexes
                    token_len = token_len.to(device)  # length of a non-padded sentence
                    word_label = word_label.to(device)
                    ability_label = ability_label.to(device)

                    word_label_out, ability_label_out = model(token_seq, token_len)
                    loss = F.binary_cross_entropy(word_label_out, word_label) + F.binary_cross_entropy(ability_label_out, ability_label)
                    loss_meter.update(loss.item(), batch_size)

                    word_pred_npy = torch.round(word_label_out).detach().numpy()
                    word_label_npy = word_label.detach().numpy()
                    ability_pred_npy = torch.round(ability_label_out).detach().numpy()
                    ability_label_npy = ability_label.detach().numpy()

                    word_label_arr.append(word_label_npy)
                    word_pred_arr.append(word_pred_npy)
                    ability_label_arr.append(ability_label_npy)
                    ability_pred_arr.append(ability_pred_npy)

                    pbar.update()


        word_label_arr = np.concatenate(word_label_arr).flatten()
        word_pred_arr = np.concatenate(word_pred_arr).flatten()
        ability_label_arr = np.concatenate(ability_label_arr).flatten()
        ability_pred_arr = np.concatenate(ability_pred_arr).flatten()

        word_acc = np.mean(word_pred_arr == word_label_arr)
        ability_acc = np.mean(ability_pred_arr == ability_label_arr)
        word_f1 = f1_score(word_label_arr, word_pred_arr)
        ability_f1 = f1_score(ability_label_arr, ability_pred_arr, average='micro')

        print('====> {} Epoch: {}\tLoss: {:.4f}\tWord Accuracy: {:.4f}\tWord F1: {:.4f}\tAbility Accuracy: {:.4f}\tAbility F1: {:.4f}'.format(
            name, epoch, loss_meter.avg, word_acc, word_f1, ability_acc, ability_f1))
        
        return loss_meter.avg, word_acc, word_f1, ability_acc, ability_f1


    best_loss = np.inf
    track_train_loss = np.zeros(config['epochs'])
    track_val_loss = np.zeros(config['epochs'])
    track_test_loss = np.zeros(config['epochs'])
    track_train_word_acc = np.zeros(config['epochs'])
    track_val_word_acc = np.zeros(config['epochs'])
    track_test_word_acc = np.zeros(config['epochs'])
    track_train_word_f1 = np.zeros(config['epochs'])
    track_val_word_f1 = np.zeros(config['epochs'])
    track_test_word_f1 = np.zeros(config['epochs'])
    track_train_ability_acc = np.zeros(config['epochs'])
    track_val_ability_acc = np.zeros(config['epochs'])
    track_test_ability_acc = np.zeros(config['epochs'])
    track_train_ability_f1 = np.zeros(config['epochs'])
    track_val_ability_f1 = np.zeros(config['epochs'])
    track_test_ability_f1 = np.zeros(config['epochs'])

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_word_acc, train_word_f1, train_ability_acc, train_ability_f1 = train(epoch)
        # we have a validation set usually to pick the best model
        val_loss, val_word_acc, val_word_f1, val_ability_acc, val_ability_f1 = test(epoch, val_loader, name='Val')
        # the test set is whats actually reported 
        test_loss, test_word_acc, test_word_f1, test_ability_acc, test_ability_f1 = test(epoch, test_loader, name='Test')
        
        track_train_loss[epoch - 1] = train_loss
        track_val_loss[epoch - 1] = val_loss
        track_test_loss[epoch - 1] = test_loss
        track_train_word_acc[epoch - 1] = train_word_acc
        track_val_word_acc[epoch - 1] = val_word_acc
        track_test_word_acc[epoch - 1] = test_word_acc
        track_train_word_f1[epoch - 1] = train_word_f1
        track_val_word_f1[epoch - 1] = val_word_f1
        track_test_word_f1[epoch - 1] = test_word_f1
        track_train_ability_acc[epoch - 1] = train_ability_acc
        track_val_ability_acc[epoch - 1] = val_ability_acc
        track_test_ability_acc[epoch - 1] = test_ability_acc
        track_train_ability_f1[epoch - 1] = train_ability_f1
        track_val_ability_f1[epoch - 1] = val_ability_f1
        track_test_ability_f1[epoch - 1] = test_ability_f1

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_lookup_path': train_lookup_path,
            'train_user_path': train_user_path,
            'text_length': len(train_dataset.words),
        }, is_best, folder=config['out_dir'])
        
        np.save(os.path.join(config['out_dir'], 'train_loss.npy'), track_train_loss)
        np.save(os.path.join(config['out_dir'], 'val_loss.npy'), track_val_loss)
        np.save(os.path.join(config['out_dir'], 'test_loss.npy'), track_test_loss)
        np.save(os.path.join(config['out_dir'], 'train_word_acc.npy'), track_train_word_acc)
        np.save(os.path.join(config['out_dir'], 'val_word_acc.npy'), track_val_word_acc)
        np.save(os.path.join(config['out_dir'], 'test_word_acc.npy'), track_test_word_acc)
        np.save(os.path.join(config['out_dir'], 'train_word_f1.npy'), track_train_word_f1)
        np.save(os.path.join(config['out_dir'], 'val_word_f1.npy'), track_val_word_f1)
        np.save(os.path.join(config['out_dir'], 'test_word_f1.npy'), track_test_word_f1)
        np.save(os.path.join(config['out_dir'], 'train_ability_acc.npy'), track_train_ability_acc)
        np.save(os.path.join(config['out_dir'], 'val_ability_acc.npy'), track_val_ability_acc)
        np.save(os.path.join(config['out_dir'], 'test_ability_acc.npy'), track_test_ability_acc)
        np.save(os.path.join(config['out_dir'], 'train_ability_f1.npy'), track_train_ability_f1)
        np.save(os.path.join(config['out_dir'], 'val_ability_f1.npy'), track_val_ability_f1)
        np.save(os.path.join(config['out_dir'], 'test_ability_f1.npy'), track_test_ability_f1)

