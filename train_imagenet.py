#!/usr/bin/python
import os
import argparse
from typing import Tuple, Dict

import scipy
import scipy.optimize
import numpy as np

from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from tensorboardX import SummaryWriter

import torchvision.transforms as transforms
from datasets.imagenet import NatImageFolder
from torchvision.datasets import ImageFolder

from core.dataloader import DataLoader
from torch.utils.data.dataloader import DataLoader as TorchDataLoader


"""
Script Hyper Parameters.
"""
use_cuda: bool = torch.cuda.is_available()
im_shape: Tuple[int] = (32, 32)
num_classes: int = 10


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


MODEL_NAMES = ['resnet']


def initialise_resnet(use_grayscale: bool=True,
                      use_im_gradients: bool=True,
                      im_shape: Tuple[int]= (32, 32),
                      num_classes: int=10):
    from models.resnet import BasicBlock, ResNetEncoder, MlpDecoder

    encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2],
                            im_grayscale=use_grayscale,
                            im_gradients=use_im_gradients)

    out_shape = encoder.get_output_shape(im_shape)
    decoder = MlpDecoder(input_shape=out_shape, num_classes=num_classes)
    return encoder, decoder


def initialise_transforms(args):

    train_transforms = [
        transforms.Scale(256),
    ]
    val_transforms = [
        transforms.Scale(256),
    ]

    if not args.im_color:
        train_transforms.append(transforms.Lambda(lambda img: img.convert('L')))
        val_transforms.append(transforms.Lambda(lambda img: img.convert('L')))

    val_transforms.append(transforms.CenterCrop(224))
    val_transforms.append(transforms.ToTensor())

    train_transforms.append(transforms.RandomCrop(224))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.ToTensor())

    # Imagenet RGB normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if not args.im_color:
        # RGB to grayscale linear map.
        t = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        mean = mean.dot(t)
        std = np.sqrt(np.square(std).dot(np.square(t)))

    normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    val_transforms.append(normalize)
    train_transforms.append(normalize)

    return train_transforms, val_transforms


def get_checkpoint(checkpt_path: str) -> Dict:
    """
    Returns the content of the checkpoint. or none if not found.
    :param checkpt_path: the checkpoint path
    :return: the checkpoint content
    """
    if not os.path.exists(checkpt_path):
        return None
    return torch.load(checkpt_path)


def restore_args(args, chckpt):
    """
    Restore the argument based on the state of the checkpoint.
    :param args: the argument object.
    :param chckpt: the checkpoint object.
    :return: threstored arguments.
    """
    # Make more robust -> Have a arg or opts class.
    # Override the dict such that can easily export to json/pickle
    # use prop name checking to ensure we're running the same list of options.
    # if not throw error. / incompatible checkpoint.
    args.arch = chckpt['args']['arch']
    args.im_color = chckpt['args']['im_color']
    args.no_gradients = chckpt['args']['no_gradients']
    args.current_epoch = chckpt['epoch']
    return args


def initialise_arg_parser():
    parser = argparse.ArgumentParser(description='Noise as Target Training')

    parser.add_argument('data_dir', type=str, help='path to dataset')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str,
                        help='path to checkpoints directory')
    parser.add_argument('--log_dir', default='./logs', type=str,
                        help='path to tensorboard log directory')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        choices=MODEL_NAMES,
                        help='model architecture: ' + ' | '.join(MODEL_NAMES) + ' (default: resnet)')

    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--current-epoch', default=0, type=int, metavar='N',
                        help='index of the first epoch')

    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ut', '--update-targets', default=3, type=int,
                        metavar='update_targets', help='frequency at which re-assign noise targets (in epochs)')
    parser.add_argument('--td', '--train-decoder', default=10, type=int,
                        metavar='train_decoder', help='frequency at which train the decoder (in epochs)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--color', dest='im_color', action='store_true',
                        help='whether to use color images instead of grayscale')

    parser.add_argument('--no_gradients', dest='no_gradients', action='store_true',
                        help='whether to use raw pixels as network input instead of image gradients.')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    return parser


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

    parser = initialise_arg_parser()
    args = parser.parse_args()

    checkpoint = None
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = get_checkpoint(args.resume)
        if checkpoint is not None:
            args = restore_args(args, checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    im_grayscale = not args.im_color
    im_gradients = not args.no_gradients

    encoder, decoder = initialise_resnet(use_grayscale=im_grayscale,
                                         use_im_gradients=im_gradients,
                                         im_shape=im_shape,
                                         num_classes=num_classes)

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    train_transforms, val_transforms = initialise_transforms(args)

    trainset = NatImageFolder(root=train_dir,
                              z_dims=encoder.get_output_shape(input_shape=im_shape),
                              transform=train_transforms)

    dl_trainset = DataLoader(trainset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.workers,
                             pin_memory=use_cuda)

    # end to end eval for encoder / decoder pair.
    val_set = ImageFolder(val_dir, transform=val_transforms)

    dl_val_set = TorchDataLoader(val_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 pin_memory=use_cuda)

    writer = SummaryWriter(log_dir=args.log_dir)

    # Freeze first layer (Compute image gradient from grayscale).
    encoder_parameters = encoder.features.parameters()

    if use_cuda:
        encoder.cuda()
        decoder.cuda()

    encoder_loss_fn = nn.MSELoss()
    decoder_loss_fn = nn.CrossEntropyLoss()

    encoder_optim = torch.optim.Adam(encoder_parameters, lr=args.lr, weight_decay=args.weight_decay)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Lr schedule in paper constant decay until t_0 then l_0 / (1+gamma*(t-t_0)+)
    # permutation every 3 epochs
    # do the freeze + learning of classifier every 20 epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optim, milestones=[10, 20], gamma=0.5)

    if use_cuda:
        decoder_loss_fn = decoder_loss_fn.cuda()
        encoder_loss_fn = encoder_loss_fn.cuda()

    best_acc = 0.0

    if checkpoint is not None:
        best_acc = checkpoint['best_acc']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        encoder_optim.load_state_dict(checkpoint['encoder_optim'])
        decoder_optim.load_state_dict(checkpoint['decoder_optim'])
        print("=> Successfully restored All model parameters. Restarting from epoch: {}".format(args.current_epoch))

    for epoch in trange(args.current_epoch, args.epochs, desc='epochs: ', leave=False):
        scheduler.step(epoch)

        update_targets = bool(((epoch+1) % args.ut) == 0)
        train_decoder = bool(((epoch+1) % args.td) == 0)

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
                    idx_step = int(((epoch+1)/args.td)*(batch_idx/100))
                    writer.add_scalar('decoder_loss', decoder_loss.data[0], idx_step)

        # Writer weight + gradient histogram for each epoch
        for name, param in encoder.named_parameters():
            name = 'encoder/'+name.replace('.', '/')
            writer.add_histogram(name, param.clone().cpu().data.numpy(), (epoch+1))
            if param.grad is not None:
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

            all_preds = np.empty(shape=(len(dl_val_set)*args.batch_size))
            all_truth = np.empty(shape=(len(dl_val_set)*args.batch_size))
            test_loss = 0.0

            for idx, (x, y) in tqdm(enumerate(dl_val_set, 0)):
                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                y_pred = decoder(encoder(x))
                loss = decoder_loss_fn(y_pred, y)
                test_loss += loss.data[0]

                init_pos = idx * args.batch_size
                all_truth[init_pos:init_pos + len(y)] = y.cpu().data.numpy()
                y_hat = np.argmax(y_pred.cpu().data.numpy(), axis=1)
                all_preds[init_pos:init_pos + len(y)] = y_hat

            # compute test stats & add to tensorboard
            all_preds = all_preds.astype(np.int32)
            all_truth = all_truth.astype(np.int32)

            # acc score
            acc_score = accuracy_score(all_truth, all_preds)
            writer.add_scalar('accuracy_score', acc_score, (epoch+1))

            if acc_score > best_acc:
                print(f'saving best encoder / decoder pair....')

                state = {
                    'args': {
                        'arch': args.arch,
                        'im_color': args.im_color,
                        'no_gradients': args.no_gradients
                    },

                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),

                    'encoder_optim': encoder_optim.state_dict(),
                    'decoder_optim': decoder_optim.state_dict(),

                    'best_acc': acc_score,
                    'epoch': epoch+1,
                }
                if not os.path.isdir(args.checkpoint_dir):
                    os.mkdir(args.checkpoint_dir)
                torch.save(state, os.path.join(args.checkpoint_dir, f'chkpt_full_{epoch}.pkl'))


if __name__ == '__main__':
    try:
        train_models()
    except KeyboardInterrupt:
        # Free up all cuda memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
