from __future__ import print_function
import argparse
import os

import chainer
from chainer import cuda
from chainer import serializers, Variable

from net import Discriminator
from net import Generator

import glob
import numpy as np

from make_midi import *

import resource
def print_memory():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    print(ru.ru_maxrss)

def main():
    parser = argparse.ArgumentParser(description='Chainer: making music with DCGAN')
    parser.add_argument('--iter', '-i', type=int, default=1000,
                        help='which model we use')
    args = parser.parse_args()
    
    xp = cuda.cupy
    
    gen = Generator(n_hidden=100, bottom_width=10, bottom_height=5)
    serializers.load_npz('result/gen_iter_{}.npz'.format(args.iter), gen)
    gen.to_gpu()
    print_memory()
    z = Variable(xp.asarray(gen.make_hidden(100)))
    with chainer.using_config('train', False):
        x = gen(z)
    x = chainer.cuda.to_cpu(x.data)
    
    arr = x2arr(x)
    arr2midi(arr)
    
if __name__ == "__main__":
    main()
    
    