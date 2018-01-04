#!/usr/bin/python
import os
import scipy
import scipy.optimize
import numpy as np

from tqdm import trange, tqdm
from sklearn.metrics import f1_score, accuracy_score


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from tensorboardX import SummaryWriter

import torchvision.transforms as transforms
from datasets.cifar_10 import NatCIFAR10, NatDataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader

from model import BasicBlock, ResNetEncoder, MlpDecoder


"""
Script Hyper Parameters.
"""
use_cuda: bool = True
checkpoint_dir = './checkpoints'

w_encoder_out: int = 512  # hidden layer dimensionality
lr: int = 1e-4
mlp_lr: int = 1e-3
batch_sz: int = 256
pin_memory: bool = False
n_epochs: int = 200

# unit: number of epochs
update_targets_frequency: int = 3
train_decoder_frequency: int = 10


# cifar classes.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# List of Dataset transforms
# CIFAR already standardized [0;1]
# use hardcoded normalisation as doesn't make much of a difference.
transform = transforms.Compose([
    transforms.ToTensor(),
    # mean / std settings from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


"""
Defines Helper functions used to train NAT Model.
"""


def calc_optimal_target_permutation(feats: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute the new target assignment that minimises the SSE between the mini-batch feature space and the targets.

    :param feats: the learnt features (given some input images)
    :param targets: the currently assigned targets.
    :return: the targets reassigned such that the SSE between features and targets is minimised for the batch.
    """
    # Compute cost matrix
    cost_matrix = np.zeros([feats.shape[0], targets.shape[0]])
    # calc SSE between all features and targets
    for i in range(feats.shape[0]):
        cost_matrix[:, i] = np.sum(np.square(feats-targets[i, :]), axis=1)

    _, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    # Permute the targets based on hungarian algorithm optimisation
    targets[range(feats.shape[0])] = targets[col_ind]
    return targets


"""
Main Encoder // Decoder Training Loop.
"""


def train_models() -> None:
    """
    # Start with smooth unit hypersphere -> We cover uniformly the space and randomly assign the features representation to the input image.
    # At every iteration we solve the credit assignment problem for each mini batch
    # We compute the loss between each input image and all the targets in the minibatch
    # Then we treat this as credit assignment problem -> we reassign the targets to the images in the mini batch s.t. distance between current representation and target
    # is minimized
    # we shuffle the dataset every epoch (Super Important)  -> Required to ensure each minibatch gets different image / target as learning goes.
    # As network learns the targets close in space should get assign to similar images.

    :return: Nothing
    """

    # create streaming for Cifar with x=latent representation and y=class labels.
    trainset = NatCIFAR10(root='./data', train=True, z_dims=w_encoder_out,
                          download=True, transform=transform)
    dl_trainset = NatDataLoader(trainset, batch_size=batch_sz,
                                shuffle=True, pin_memory=pin_memory)

    # end to end eval for encoder / decoder pair.
    val_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    dl_val_set = DataLoader(val_set, batch_size=batch_sz, shuffle=False, pin_memory=pin_memory)

    writer = SummaryWriter()

    encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2])
    # todo: remove hard-coded param for model param extract. & also in NAT Cifar z_dim.
    decoder = MlpDecoder(w_encoder_out, num_classes=10)

    if use_cuda:
        encoder.cuda()
        decoder.cuda()

    encoder_loss_fn = nn.MSELoss()
    decoder_loss_fn = nn.CrossEntropyLoss()

    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-3)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=mlp_lr, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optim, milestones=[10, 20], gamma=0.5)

    if use_cuda:
        decoder_loss_fn = decoder_loss_fn.cuda()
        encoder_loss_fn = encoder_loss_fn.cuda()

    best_acc = 0.0

    for epoch in trange(n_epochs, desc='epochs: ', leave=False):
        scheduler.step(epoch)
        for group in encoder_optim.param_groups:
            print(group['lr'])

        update_targets = bool(((epoch+1) % update_targets_frequency) == 0)
        train_decoder = bool(((epoch+1) % train_decoder_frequency) == 0)

        if update_targets:
            print(f'Train model & update targets for epoch: {epoch}.')
        else:
            print(f'Train model for epoch {epoch}.')

        # set encoder to training mode. (for batch norm layers)
        encoder.train(True)
        decoder.train(train_decoder)

        # Stream Training dataset with NAT
        for batch_idx, (idx, x, y, nat) in enumerate(tqdm(dl_trainset, 0), 1):
            e_targets = nat.numpy()
            if use_cuda:
                x, y, nat = x.cuda(), y.cuda(), nat.cuda()

            x = Variable(x)
            encoder_optim.zero_grad()
            outputs = encoder(x)

            # every few iterations greedy re-assign targets.
            if update_targets:
                e_out = outputs.cpu().data.numpy()
                new_targets = calc_optimal_target_permutation(e_out, e_targets)
                # update.
                trainset.update_targets(idx, new_targets)
                nat = torch.FloatTensor(new_targets)
                if use_cuda:
                    nat = nat.cuda()

            # train encoder
            nat = Variable(nat)
            encoder_loss = encoder_loss_fn(outputs, nat)
            encoder_loss.backward(retain_graph=True)
            encoder_optim.step()

            if batch_idx % 100 == 0:
                writer.add_scalar('encoder_loss', encoder_loss.data[0], int((epoch+1)*(batch_idx/100)))

            if train_decoder:
                y = Variable(y)
                decoder_optim.zero_grad()
                y_pred = decoder(outputs)
                decoder_loss = decoder_loss_fn(y_pred, y)
                decoder_loss.backward()
                decoder_optim.step()

                if batch_idx % 100 == 0:
                    idx_step = int(((epoch+1)/train_decoder_frequency)*(batch_idx/100))
                    writer.add_scalar('decoder_loss', decoder_loss.data[0], idx_step)

        # Writer weight + gradient histogram for each epoch
        for name, param in encoder.named_parameters():
            name = 'encoder/'+name.replace('.', '/')
            writer.add_histogram(name, param.clone().cpu().data.numpy(), (epoch+1))
            writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), (epoch+1))

        # if decoder has been trained, eval classifier
        if train_decoder:

            # write decoder weight + gradient once trained for 1 epoch.
            for name, param in decoder.named_parameters():
                name = 'decoder/' + name.replace('.', '/')
                writer.add_histogram(name, param.clone().cpu().data.numpy(), (epoch+1))
                writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), (epoch+1))

            # set models to eval mode and validate on test set.
            encoder.eval()
            decoder.eval()

            all_preds = np.empty(shape=(len(dl_val_set)*batch_sz))
            all_truth = np.empty(shape=(len(dl_val_set)*batch_sz))
            test_loss = 0.0

            for idx, (x, y) in tqdm(enumerate(dl_val_set, 0)):
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                y_pred = decoder(encoder(x))
                loss = decoder_loss_fn(y_pred, y)
                test_loss += loss.data[0]

                init_pos = idx * batch_sz
                all_truth[init_pos:init_pos + len(y)] = y.cpu().data.numpy()
                y_hat = np.argmax(y_pred.cpu().data.numpy(), axis=1)
                all_preds[init_pos:init_pos + len(y)] = y_hat

            # compute test stats & add to tensorboard

            all_preds = all_preds.astype(np.int32)
            all_truth = all_truth.astype(np.int32)

            # acc score / f1 score (micro / macro)
            acc_score = accuracy_score(all_truth, all_preds)
            f1_micro = f1_score(all_truth, all_preds, labels=classes, average='micro')
            f1_macro = f1_score(all_truth, all_preds, labels=classes, average='macro')

            writer.add_scalar('accuracy_score', acc_score, (epoch+1))
            writer.add_scalar('f1 micro', f1_micro, (epoch+1))
            writer.add_scalar('f1 macro', f1_macro, (epoch+1))

            if acc_score > best_acc:
                print(f'saving best encoder / decoder pair....')

                state = {
                    'encoder': encoder if use_cuda else encoder,
                    'decoder': decoder if use_cuda else decoder,
                    'acc': acc_score,
                    'epoch': (epoch+1),
                }
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                torch.save(state, os.path.join(checkpoint_dir, f'chkpt_full_{epoch}.pkl'))


if __name__ == '__main__':
    try:
        train_models()
    except KeyboardInterrupt:
        # Free up all cuda memory
        torch.cuda.empty_cache()
