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
from data.Air_helper import map_web_to_source


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Weak-shot Learning')
    parser.add_argument('--data_path', default='../dataset/CUB', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--type', default='hardest', type=str)
    parser.add_argument('--imagenet_pretrained', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--exp_name', default='duplicate', type=str)

    return parser.parse_args()


def main():
    args = get_parser()

    args.exp_name += f'_{os.path.basename(args.data_path)}'

    init_log(args.exp_name)
    log(args)

    data_helper = get_data_helper(args)

    save_path = f'pretrained/Duplicate/{data_helper.name}/meta'
    os.makedirs(save_path, exist_ok=True)

    train_set = data_helper.get_noisy_novel_loader().dataset
    test_set = data_helper.get_novel_test_loader().dataset

    # test_set.get_dict()
    eval_transforms = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_set.transform = eval_transforms
    test_set.transform = eval_transforms

    train_loader = Loader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = Loader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = nn.DataParallel(
        Backbone(mhelper.get_model(args, train_loader.dataset.num_classes()))
    ).cuda()
    model.eval()

    test_names, test_feats = get_test_names_and_features(save_path, model, test_loader)
    train_names, train_min_dist_values, train_nearest_test_names = get_train_meta(
        save_path, model, train_loader, test_names, test_feats)

    meta_dict = {}
    categories = train_loader.dataset.categories
    for category in train_loader.dataset.categories:
        category = map_web_to_source(category, list(test_names.keys()))

        meta_dict[category] = [[], [], []]

    for train_name, value, test_name in zip(
            train_names, train_min_dist_values, train_nearest_test_names):
        # category = test_name.split('/')[-2]
        # our_path = test_name
        # if our_path.split('/')[-2] in categories:
        #     category = our_path.split('/')[-2]
        #
        # elif our_path.split('/')[-3] + '_' + our_path.split('/')[-2] in categories:
        #     category = our_path.split('/')[-3] + '_' + our_path.split('/')[-2]

        category = map_web_to_source(train_name.split('/')[-2], list(test_names.keys()))

        meta_dict[category][0].append(train_name)
        meta_dict[category][1].append(value)
        meta_dict[category][2].append(test_name)

    remain_dict = {}
    for category, (train_names, values, test_names) in meta_dict.items():
        sorted_args = np.argsort(values)
        if args.type == 'hardest':
            hard_args = sorted_args[-1000:]
        elif args.type == 'easiest':
            hard_args = sorted_args[:1000]
        else:
            raise NotImplementedError

        remain_dict[category] = [train_names[i] for i in hard_args]

        # show(train_names[sored_args[0]], test_names[sored_args[0]], values[sored_args[0]])
    torch.save(remain_dict, f'{save_path}/remain_dict.pth')

    make_dataset(remain_dict, args.type)
    return


def make_dataset(remain_dict, type):
    new_root = f'../dataset/{type}_new_set'
    from utils.project_kits import copy_to
    for category, source_list in tqdm(remain_dict.items()):
        for i, spath in enumerate(np.random.permutation(source_list)):
            copy_to(spath, f'{new_root}/{category}/{i}_{os.path.basename(spath)}')
    return


def get_train_meta(save_path, model, train_loader, test_names, test_feats):
    if not os.path.exists(f'{save_path}/train_nearest_test_names.pth'):
        with torch.no_grad():
            print('get train_meta')
            log(f'total steps: {len(train_loader)}')
            train_names = []
            train_min_dist_values = []
            train_nearest_test_names = []

            for batch_i, (images, categories, im_names) in tqdm(enumerate(train_loader)):
                train_feats = model(images.cuda()).data.cpu()

                for i, (feat, cid, name) in enumerate(zip(train_feats, categories, im_names)):
                    category = train_loader.dataset.int2category[cid.item()]
                    category = map_web_to_source(category, list(test_names.keys()))
                    test_feat_c = test_feats[category]
                    distance = edu_distance(feat.unsqueeze(0), test_feat_c)

                    min_dist_value, nearest_test_id = distance.min(1)
                    train_names.append(name)
                    train_min_dist_values.append(min_dist_value.item())
                    train_nearest_test_names.append(test_names[category][nearest_test_id])

        torch.save(train_names, f'{save_path}/train_names.pth')
        torch.save(train_min_dist_values, f'{save_path}/train_min_dist_values.pth')
        torch.save(train_nearest_test_names, f'{save_path}/train_nearest_test_names.pth')
    else:
        print('load train_meta')
        train_names = torch.load(f'{save_path}/train_names.pth')
        train_min_dist_values = torch.load(f'{save_path}/train_min_dist_values.pth')
        train_nearest_test_names = torch.load(f'{save_path}/train_nearest_test_names.pth')

    return train_names, train_min_dist_values, train_nearest_test_names


def get_test_names_and_features(save_path, model, test_loader):
    if not os.path.exists(f'{save_path}/test_feats.pth'):
        print('get test_names_and_features')
        test_names, test_feats = {}, {}

        for c in test_loader.dataset.categories:
            test_names[c], test_feats[c] = [], []

        with torch.no_grad():
            for batch_i, (images, categories, im_names) in tqdm(enumerate(test_loader)):
                feats = model(images.cuda()).data.cpu()

                for feat, cid, name in zip(feats, categories, im_names):
                    test_names[test_loader.dataset.int2category[cid.item()]].append(name)
                    test_feats[test_loader.dataset.int2category[cid.item()]].append(feat)

        for c in test_loader.dataset.categories:
            test_feats[c] = torch.stack(test_feats[c])

        torch.save(test_names, f'{save_path}/test_names.pth')
        torch.save(test_feats, f'{save_path}/test_feats.pth')
    else:
        print('load test_names_and_features')
        test_names = torch.load(f'{save_path}/test_names.pth')
        test_feats = torch.load(f'{save_path}/test_feats.pth')

    return test_names, test_feats


def edu_distance(P, Q):
    (P.unsqueeze(1) - Q.unsqueeze(0))
    return ((P.unsqueeze(1) - Q.unsqueeze(0)) ** 2).sum(2).sqrt()


def show(p1, p2, s):
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211)
    ax1.imshow(plt.imread(p1))
    ax1.set_title(p1)

    ax2 = plt.subplot(212)
    ax2.imshow(plt.imread(p2))
    ax2.set_title(p2)

    plt.suptitle(f's:{s}')
    plt.show()
    plt.close()
    return


# def local():
#     args = get_parser()
#
#     train_names = torch.load('pretrained/duplicate_Dog/train_names.pth')
#     test_names = torch.load('pretrained/duplicate_Dog/test_names.pth')
#     train_min_dists = torch.cat(torch.load('pretrained/duplicate_Dog/train_min_dist.pth'))
#     train_min_ids = torch.cat(torch.load('pretrained/duplicate_Dog/train_min_id.pth'))
#
#     def show(p1, p2, s, rank):
#         import matplotlib.pyplot as plt
#         ax1 = plt.subplot(211)
#         ax1.imshow(plt.imread(p1))
#         ax1.set_title(p1)
#
#         ax2 = plt.subplot(212)
#         ax2.imshow(plt.imread(p2))
#         ax2.set_title(p2)
#
#         plt.suptitle(f's:{s}')
#         plt.savefig(f'pretrained/{rank}.jpg')
#         plt.close()
#         return
#
#     selected_train_id = train_min_dists.sort()[1][:1000]
#     selected_test_id = train_min_ids[selected_train_id]
#
#     X = []
#     Y = []
#     for i, (p, q) in tqdm(enumerate(zip(selected_train_id, selected_test_id))):
#         dis = train_min_dists[p]
#         if dis < 9:
#             show(train_names[p], test_names[q], dis, i)
#             X.append(train_names[p])
#             Y.append(test_names[q])
#
#     torch.save(X, f'pretrained/OurGoogleDog28_trainset_duplicates_X.pth')
#     torch.save(Y, f'pretrained/OurGoogleDog28_trainset_duplicates_Y.pth')
#     return


class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return self.backbone.avgpool(x).squeeze(2).squeeze(2)


if __name__ == '__main__':
    main()
