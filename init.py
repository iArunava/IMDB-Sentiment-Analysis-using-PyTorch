import torch
import numpy as np
import argparse
import subprocess
from train import train
from test import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
        type=str,
        default='./dataset/',
        help='The path to the dataset directory')

    parser.add_argument('-t', '--train',
        type=int,
        default=0,
        help='0 says test and any other number says train. Default Test')

    parser.add_argument('-r', '--review',
        type=str,
        help='The review to be tested')

    parser.add_argument('-ed', '--embedding-dim',
        type=int,
        default=400,
        help='The embedding dimension')

    parser.add_argument('-hd', '--hidden-dim',
        type=int,
        default=512,
        help='The hidden dimension')

    parser.add_argument('-nl', '--n-layers',
        type=int,
        default=2,
        help='The number of layers in the RNN')

    parser.add_argument('-lr', '--learning-rate',
        type=float,
        default=0.01,
        help='The learning rate')

    parser.add_argument('-e', '--epochs',
        type=int,
        default=4,
        help='The number of epochs')

    parser.add_argument('-pe', '--print-every',
        type=int,
        default=500,
        help='The interval after which the loss and accuracy stats are to be \
                printed during the training')

    parser.add_argument('-sl', '--seq-length',
        type=int,
        default=200,
        help='The sequence length up to which each review is to be padded')

    parser.add_argument('-sf', '--split-frac',
        type=int,
        default=0.8,
        help='The fraction for the training set')

    parser.add_argument('-bs', '--batch-size',
        type=int,
        default=16,
        help='The batch size')

    parser.add_argument('-c', '--clip',
        type=float,
        default=5.0,
        help='The number at which the gradients will be clipped')

    FLAGS, unparsed = parser.parse_known_args()

    # Call function based on train or test
    if (FLAGS.train == 0):
        if (FLAGS.review == '' or FLAGS.review is None):
            raise Exception('[ERROR]Specify a review with the -r option that needs to be tested')
        print ('[INFO]Test Review Found!!')

        net = load_model(model_path)
        predict(net, FLAGS.review)
    else:
        train(FLAGS)

    print ('[INFO]Exiting...')
