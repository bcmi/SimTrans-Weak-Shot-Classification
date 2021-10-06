import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import Sampler
from torchvision import models

Norm = nn.BatchNorm1d


class DomainClassifier(nn.Module):
    def __init__(self, args, diff_dim=2048):
        super(DomainClassifier, self).__init__()

        self.fc = nn.Linear(diff_dim, diff_dim)

        if args.domain_bn == 1:
            self.bn = nn.LayerNorm(diff_dim)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(diff_dim, 2)

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def pair_enumeration(x):
    '''
        input:  [B,D]
        return: [B*B,D]

        input  [[a],
                [b]]
        return [[a,a],
                [b,a],
                [a,b],
                [b,b]]
    '''
    assert x.ndimension() == 2, 'Input dimension must be 2'
    # [a,b,c,a,b,c,a,b,c]
    # [a,a,a,b,b,b,c,c,c]
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return torch.cat((x1, x2), dim=1)


def _pair_enumeration(x):
    '''
        input:  [B,D]
        return: [B*B,D]

        input  [[a],
                [b]]
        return [[a,a],
                [b,a],
                [a,b],
                [b,b]]
    '''
    assert x.ndimension() == 2, 'Input dimension must be 2'
    # [a,b,c,a,b,c,a,b,c]
    # [a,a,a,b,b,b,c,c,c]
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2


class Backbone(nn.Module):
    def __init__(self, pretrained):
        super(Backbone, self).__init__()
        if pretrained == 'none':
            print(f'no pretrained')
            self.resnet = models.resnet50(pretrained=False)
        elif pretrained == 'imagenet':
            print(f'ImageNet pretrained')
            self.resnet = models.resnet50(pretrained=True)
        else:
            print(f'pretrained: {pretrained}')
            self.resnet = torch.load(pretrained).module

        self.n_feat = 2048
        del self.resnet.fc

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class GANSimilarityHead4(nn.Module):
    def __init__(self, n_feat=512):
        super(GANSimilarityHead4, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat * 2),
            Norm(n_feat * 2),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat),
            Norm(n_feat),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(n_feat, n_feat),
            Norm(n_feat),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Linear(n_feat, 2)

        self.diff_dim = n_feat

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        diff_feat = x

        x = self.fc3(x)
        x = self.fc4(x)

        return x, diff_feat


class GANSimilarityHead2(nn.Module):
    def __init__(self, n_feat=512):
        super(GANSimilarityHead2, self).__init__()

        self.fc = nn.Linear(n_feat * 2, n_feat * 4)
        self.bn = Norm(n_feat * 4)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(n_feat * 4, 2)

        self.diff_dim = n_feat * 4

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        diff_feat = x
        pred = self.fc2(x)
        return pred, diff_feat


class GANSimilarityNet(nn.Module):
    def __init__(self, args):
        head_type = args.head_type
        pretrained = args.similarity_pretrained

        super(GANSimilarityNet, self).__init__()
        self.backbone = nn.DataParallel(Backbone(pretrained))

        self.similarity_head = nn.DataParallel(
            eval(f'GANSimilarityHead{head_type}')(n_feat=self.backbone.module.n_feat))
        self.diff_dim = self.similarity_head.module.diff_dim

    def reset_gpu(self):
        self.backbone = nn.DataParallel(self.backbone.module)
        self.similarity_head = nn.DataParallel(self.similarity_head.module)

    def forward(self, images):
        '''
            the [:,1] is for similarity, and [:,0] dissimilarity.
        '''
        features = self.backbone(images)
        feature_pairs = pair_enumeration(features)
        similarities, diff_feat = self.similarity_head(feature_pairs)

        if self.training:
            return similarities, diff_feat
        else:
            return similarities


def make_similarities(categories):
    pair = pair_enumeration(categories.unsqueeze(1))
    return (pair[:, 0] == pair[:, 1]).long()


class RandPairSet(Dataset):
    def __init__(self, dataset, class_num_per, batch_size):
        self.dataset = dataset
        self.class_num_per = class_num_per
        self.batch_size = batch_size
        self.pairs_init()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ids = self.pairs[index]
        imgs, cs, ps = [], [], []
        for i in ids:
            img, c, p = self.dataset.__getitem__(i)
            imgs.append(img)
            cs.append(c)
            ps.append(p)

        return torch.stack(imgs), torch.tensor(cs), ps

    def pairs_init(self):
        pairs = []

        data_dict = {}
        for i, (p, c) in enumerate(self.dataset.image_list):
            if c in data_dict.keys():
                data_dict[c].append(i)
            else:
                data_dict[c] = [i]

        length = len(self.dataset) // self.batch_size

        for i in range(length):
            batch = []

            # 查找有剩余的类别
            remained_categories = []
            for k, v in data_dict.items():
                if len(v) > 0:
                    remained_categories.append(k)

            # 如果都没有剩余，就结束
            if not remained_categories:
                break

            # 选择从哪些类中抽样
            selected_categories = np.random.permutation(remained_categories).tolist()[:self.class_num_per]

            for j in range(self.batch_size):
                # 选择从哪类中抽当前样本
                if not selected_categories:
                    break

                selected = np.random.randint(0, len(selected_categories))
                selected_c = selected_categories[selected]
                cur_length = len(data_dict[selected_c])

                batch.append(data_dict[selected_c].pop(np.random.randint(cur_length)))
                if cur_length == 1:
                    selected_categories.pop(selected)

            pairs.append(batch)

        self.pairs = pairs


class DiffPairSet(Dataset):
    def __init__(self, dataset, class_num_per, batch_size, max_length=5000):
        self.dataset = dataset
        self.class_num_per = class_num_per
        self.batch_size = batch_size
        self.max_length = max_length
        self.pairs_init()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ids = self.pairs[index]
        imgs, cs, ps = [], [], []
        for i in ids:
            img, c, p = self.dataset.__getitem__(i[0])
            imgs.append(img)
            cs.append(c)
            ps.append(p)

        return torch.stack(imgs), torch.tensor(cs), ps

    def pairs_init(self):
        pairs = []
        data_dict = {}
        for i, (p, c) in enumerate(self.dataset.image_list):
            if c in data_dict.keys():
                data_dict[c].append(i)
            else:
                data_dict[c] = [i]
        class_num = len(data_dict.keys())
        for i in range(self.max_length):
            batch = []
            selected_categories = np.random.choice(list(data_dict.keys()), min(self.batch_size, class_num),
                                                   replace=False)
            for c in selected_categories:
                batch.append(np.random.choice(data_dict[c], 1))

            pairs.append(batch)
        self.pairs = pairs


class FuseSet(Dataset):
    def __init__(self, pair_set1, pair_set2):
        self.pair_set1 = pair_set1
        self.pair_set2 = pair_set2

    def __len__(self):
        return len(self.pair_set1) + len(self.pair_set2)

    def __getitem__(self, index):
        L1 = len(self.pair_set1)
        if index < L1:
            return self.pair_set1.__getitem__(index)
        else:
            return self.pair_set2.__getitem__(index - L1)


def get_train_loader(args, train_loader):
    train_pairs = RandPairSet(train_loader.dataset, args.batch_class_num, args.batch_size)
    return DataLoader(train_pairs, batch_size=1, shuffle=True)


def get_train_loader_(args, train_loader, noisy_loader, batch_class_num):
    train_pairs = RandPairSet(train_loader.dataset, batch_class_num, args.batch_size)
    noisy_pairs = DiffPairSet(noisy_loader.dataset, batch_class_num, args.batch_size,
                              max_length=int(len(train_pairs) * args.noisy_frac))

    set = FuseSet(train_pairs, noisy_pairs)
    return DataLoader(set, batch_size=1, shuffle=True)


def get_similarity_test_loader(args, data_loader, class_num_per_batch=10, sample_num_per_class=10):
    if args.batch_class_num > 0:
        return DataLoader(data_loader.dataset,
                          batch_sampler=SequentialBalancedSampler(
                              data_loader.dataset, class_num_per_batch=class_num_per_batch,
                              sample_num_per_class=sample_num_per_class, drop=True),
                          batch_size=0, shuffle=None,
                          drop_last=None, sampler=None,
                          num_workers=args.num_workers)
    else:
        return data_loader


class SequentialBalancedSampler(Sampler):
    def __init__(self, dataset, class_num_per_batch, sample_num_per_class, drop=True):
        if not drop:
            raise NotImplementedError

        image_dict = {}
        for k in dataset.int2category.keys():
            image_dict[k] = []
        for dataset_idx, (p, c) in enumerate(dataset.image_list):
            image_dict[c].append(dataset_idx)

        idx_lists = []

        i = 0
        while min([len(x) for x in image_dict.values()]) > sample_num_per_class:
            batch = []
            for c in range(i, i + class_num_per_batch):
                selected_dict = image_dict[c % dataset.num_classes()]
                if len(selected_dict) < sample_num_per_class:
                    break
                for j in range(sample_num_per_class):
                    batch.append(selected_dict.pop())
            idx_lists.append(batch)
            i += class_num_per_batch
        self.list = idx_lists
        self.length = len(self.list)

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        self.length


class RandomBalancedSampler(Sampler):
    def __init__(self, batch_class_num, dataset, batch_size, drop_last):

        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            self.length = len(dataset) // self.batch_size
        else:
            self.length = (len(dataset) + self.batch_size - 1) // self.batch_size

        image_dict = {}
        for k in dataset.int2category.keys():
            image_dict[k] = []
        for dataset_idx, (p, c) in enumerate(dataset.image_list):
            image_dict[c].append(dataset_idx)

        idx_lists = []
        for i in range(self.length):
            batch = []

            # 查找有剩余的类别
            remained_categories = []
            for k, v in image_dict.items():
                if len(v) > 0:
                    remained_categories.append(k)

            # 如果都没有剩余，就结束
            if not remained_categories:
                break

            # 选择从哪些类中抽样
            selected_categories = np.random.permutation(remained_categories).tolist()[:batch_class_num]

            for j in range(self.batch_size):
                # 选择从哪类中抽当前样本
                if not selected_categories:
                    break

                selected = np.random.randint(0, len(selected_categories))
                selected_c = selected_categories[selected]
                cur_length = len(image_dict[selected_c])

                batch.append(image_dict[selected_c].pop(np.random.randint(cur_length)))
                if cur_length == 1:
                    selected_categories.pop(selected)

            idx_lists.append(batch)

        self.list = idx_lists

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        self.length

def get_similarity_test_loader2(args, data_loader):
    if args.batch_class_num > 0:
        return DataLoader(data_loader.dataset,
                          batch_sampler=SequentialBalancedSampler2(
                              data_loader.dataset, class_num_per_batch=10, sample_num_per_class=10, drop=True),
                          batch_size=1, shuffle=None,
                          drop_last=None, sampler=None,
                          num_workers=args.num_workers)
    else:
        return data_loader


class SequentialBalancedSampler2(Sampler):
    def __init__(self, dataset, class_num_per_batch, sample_num_per_class, drop=True, total_iter=10):
        if not drop:
            raise NotImplementedError

        image_dict = {}
        for k in dataset.int2category.keys():
            image_dict[k] = []
        for dataset_idx, (p, c) in enumerate(dataset.image_list):
            image_dict[c].append(dataset_idx)

        C = len(image_dict)
        self.list = []

        for iter in range(total_iter):
            batch = []
            selected_class_idx = np.random.choice(C, class_num_per_batch, replace=False)
            for idx in selected_class_idx:
                category_images = image_dict[idx]
                L = len(category_images)
                replace = L < sample_num_per_class
                selected_images = np.random.choice(category_images, sample_num_per_class, replace=replace).tolist()
                batch += selected_images
            self.list.append(batch)

        self.length = len(self.list)

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        self.length
