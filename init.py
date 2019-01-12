import torch
import numpy as np
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
        type=str,
        default='./dataset/'
        help='The path to the dataset directory')

    FLAGS, unparsed = parser.parse_known_args()

    # download the files if needed
    if os.path.exists(FLAGS.dataset + 'labels.txt') and
        os.path.exists(FLAGS.dataset + 'reviews.txt'):
        subprocess.call(['./dataset/download.sh'])
