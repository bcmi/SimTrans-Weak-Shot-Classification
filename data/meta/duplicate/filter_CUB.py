import sys
import warnings

sys.path.append(".")
warnings.filterwarnings("ignore")

import numpy as np
import torch.nn as nn
from tqdm import tqdm
from data.factory import get_data_helper
import models.helper as mhelper

from utils.project_kits import init_log, log

import argparse
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader as Loader
import os


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Weak-shot Learning')
    parser.add_argument('--data_path', default='../dataset/CUB', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--imagenet_pretrained', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--max_noisy_images_per', default=None, type=int)
    parser.add_argument('--exp_name', default='duplicate', type=str)

    return parser.parse_args()

def _get_web_name(category_name):
    if category_name == 'n02086240-Shih-Tzu':
        return 'Tzu'
    else:
        return (category_name.split('.')[1] + '_bird').replace('_', '+')


def _get_source_name(source_name, categories):
    if source_name == 'Tzu':
        return 'n02086240-Shih-Tzu'
    else:
        t = source_name.replace('+', '_').replace('_bird', '')
        for c in categories:
            if t in c:
                return c


def local():
    args = get_parser()
    path = 'pretrained/duplicate_CUB'

    test_name_dict = {}

    data_helper = get_data_helper(args)
    train_set = data_helper.get_all_noisy_loader().dataset.int2category
    test_set = data_helper.get_all_test_loader().dataset
    for c in test_set.categories:
        test_name_dict[c] = []

    for p, cid in test_set.image_list:
        test_name_dict[test_set.int2category[cid]].append(p)

    infof = []
    train_min_dists = []
    train_min_ids = []
    train_names = []
    test_names = []

    for f in os.listdir(path):
        if 'min_dist' in f:
            infof.append(f)

    for f in tqdm(infof):
        min_dist = torch.load(f'{path}/{f}')
        min_id = torch.load(f'{path}/{f.replace("min_dist", "min_id")}')
        train_name = torch.load(f'{path}/{f.replace("min_dist", "im_names")}')
        i = int(f.split('_')[-1].split('.')[0])

        train_min_dists.append(min_dist)
        train_min_ids.append(min_id)
        train_names.append(train_name[i])
        source_name = _get_source_name(train_name[i].split('/')[-2], test_set.categories)
        test_names.append(test_name_dict[source_name])

    train_min_dists = torch.cat(train_min_dists)
    train_min_ids = torch.cat(train_min_ids)

    def show(p1, p2, s, rank):
        import matplotlib.pyplot as plt
        ax1 = plt.subplot(211)
        ax1.imshow(plt.imread(p1))
        ax1.set_title(p1)

        ax2 = plt.subplot(212)
        ax2.imshow(plt.imread(p2))
        ax2.set_title(p2)

        plt.suptitle(f's:{s}')
        plt.savefig(f'pretrained/{rank}.jpg')
        plt.close()
        return

    selected_train_id = train_min_dists.sort()[1][:2000]
    selected_test_id = train_min_ids[selected_train_id]

    X = []
    Y = []
    for i, (p, q) in tqdm(enumerate(zip(selected_train_id, selected_test_id))):
        dis = train_min_dists[p]
        train_name = train_names[p]
        test_name = test_names[p][q]
        if dis < 9:
            show(train_name, test_name, dis, i)
            X.append(train_name)
            Y.append(test_name)

    torch.save(X, f'pretrained/OurGoogleCUB200_trainset_duplicates_X.pth')
    torch.save(Y, f'pretrained/OurGoogleCUB200_trainset_duplicates_Y.pth')
    return


if __name__ == '__main__':
    local()
