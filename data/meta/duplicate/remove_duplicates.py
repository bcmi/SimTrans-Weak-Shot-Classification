import sys
import warnings

sys.path.append(".")
warnings.filterwarnings("ignore")

import argparse
import torch
import numpy as np
import os


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Weak-shot Learning')
    parser.add_argument('--data_path', default='../dataset/CUB', type=str)

    return parser.parse_args()


def main():
    args = get_parser()
    if 'Dog' in args.data_path:
        infof = 'data/meta/duplicate/info/OurGoogleDog28_trainset_duplicates_X.pth'
    elif 'CUB' in args.data_path:
        infof = 'data/meta/duplicate/info/OurGoogleCUB200_trainset_duplicates_X.pth'
    else:
        raise NotImplementedError

    info = torch.load(infof)
    for i in info:
        try:
            os.remove(i)
        except:
            print(f'file already removed: {i}')
    return


if __name__ == '__main__':
    main()
