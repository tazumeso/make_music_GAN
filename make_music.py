#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda

from net import Discriminator
from net import Generator
from updater import DCGANUpdater
from visualize import out_generated_image

import glob
import numpy as np


def make_image(f, len_mea):
    music_list = np.load(f) * 255
    music_list = music_list.transpose(0, 2, 1, 3).tolist()
    music_image_list = []
    for i in range(5, len(music_list) - len_mea + 1):
        music_image = music_list[i]
        for j in range(1, len_mea):
            music_image += music_list[i + j]
        music_image = np.array(music_image)
        if music_image.sum() > 11:
            music_image_list.append(music_image)
    return music_image_list


def load_train(dir_name, gpu):
    input_data = []
    for f in glob.glob(dir_name + "/*"):
        input_data += make_image(f, 5)
    if gpu >= 0:
        xp = cuda.cupy
        return xp.array(input_data, dtype=xp.float32).transpose(0, 2, 1, 3)
        #return chainer.cuda.to_gpu(np.array(input_data).transpose(0, 2, 1, 3).astype(np.float32))
    return np.array(input_data).transpose(0, 2, 1, 3).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Chainer: making music with DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden, bottom_width=10, bottom_height=5)
    dis = Discriminator(bottom_width=10, bottom_height=5)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    train = load_train("music_numpy", args.gpu)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Set up a trainer
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
