import torch
import numpy as np
import argparse
import subprocess
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
        type=str,
        default='./dataset/'
        help='The path to the dataset directory')

    parser.add_argument('-t', '--train',
        type=int,
        default=0,
        help='0 says test and any other number says train. Default Test')

    parser.add_argument('-r', '--review',
        type=str,
        help='The review to be tested')

    FLAGS, unparsed = parser.parse_known_args()

    # Call function based on train or test
    if (FLAGS.train == 0):
        if (FLAGS.review == '' or FLAGS.review is None):
            raise Exception('[ERROR]Specify a review with the -r option that needs to be tested')
        print ('[INFO]Test Review Found!!')
        test()
    else:
        train()
