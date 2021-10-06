import sys

sys.path.append(".")

import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

import utils.saver as saver
from data.factory import get_data_helper

from utils.meters import MetrixMeter, AverageMeter
from utils.parser import get_base_parser, add_base_train
from utils.project_kits import init_log, log, set_seeds, occupy
from utils.vis import vis_acc
from ad_similarity.ad_modules import *
from PIL import Image
from torchvision import transforms


def add_detection(parser):
    parser.add_argument('--save_dir', type=str, default='saves/h4f2_86.5')
    parser.add_argument('--similarity_net', type=str, default='pretrained/CUB/step1/h4f2_86.5.pth')
    return parser


def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_detection(parser)
    args = parser.parse_args()

    args.exp_name += f'NoiseDetection_{os.path.basename(args.data_path)}' \
                     f'_{datetime.datetime.now().strftime("%m%d%H%M")}'
    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)
    set_seeds(args.seed)

    data_helper = get_data_helper(args)
    log(data_helper)

    noisy_set = data_helper._get_noisy_novel_set()

    image_dict = {}
    for i, c in noisy_set.image_list:
        if c in image_dict.keys():
            image_dict[c].append(i)
        else:
            image_dict[c] = [i]

    test_transforms = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    model = torch.load(args.similarity_net)
    model.reset_gpu()

    batch_size = 100
    model.eval()

    save_path = f'{args.save_dir}'
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for c, image_list in image_dict.items():
            category_name = noisy_set.int2category[c]
            print(category_name)
            batch_step = int(np.ceil(len(image_list) / batch_size))
            if not os.path.exists(f'{save_path}/{noisy_set.int2category[c]}_name.pth'):

                names = []
                feats = []
                for b in range(batch_step):
                    paths = image_list[b * batch_size:b * batch_size + batch_size]
                    images = [test_transforms(Image.open(path).convert('RGB')) for path in paths]
                    feat = model.backbone(torch.stack(images).cuda())

                    feats.append(feat.data.cpu())
                    names += paths
                feats = torch.cat(feats)
                p = f'{save_path}/{noisy_set.int2category[c]}_feat.pth'
                os.makedirs(os.path.dirname(p), exist_ok=True)
                torch.save(feats, p)
                torch.save(names, f'{save_path}/{noisy_set.int2category[c]}_name.pth')
            else:
                names = torch.load(f'{save_path}/{noisy_set.int2category[c]}_name.pth')
                feats = torch.load(f'{save_path}/{noisy_set.int2category[c]}_feat.pth')

            if not os.path.exists(f'{save_path}/{noisy_set.int2category[c]}_matrix.pth'):
                L = len(feats)
                similarity_matrix = feats.new_ones(L, L) * -1
                for i in tqdm(range(L)):
                    paired = pair_feature(p=feats[i], Q=feats)
                    res, _ = model.similarity_head(paired.cuda())
                    res = res.cpu()
                    similarity = torch.softmax(res, dim=1)[:, 1]
                    similarity_matrix[i] = similarity
                torch.save(similarity_matrix, f'{save_path}/{noisy_set.int2category[c]}_matrix.pth')
            else:
                similarity_matrix = torch.load(f'{save_path}/{noisy_set.int2category[c]}_matrix.pth')
                densities = get_density2(distance=1 - similarity_matrix)
                # if c < 2:
                #     vis_density(densities, names, category_name)
                # d = 1
    return


def vis_density(density, name, category_name):
    from utils.project_kits import copy_to
    clean_rank = density.sort()[1]
    for rank, i in tqdm(enumerate(clean_rank)):
        source_image_path = name[i]
        target_image_path = f'pretrained/vis_rank/{category_name}/{rank}.jpg'
        copy_to(source_image_path, target_image_path)
    return


def get_density2(distance):
    return (1 - distance).mean(1)


def get_density(distance, density_t=0.6):
    L = len(distance)
    densities = distance.new_zeros(L)
    flat_distance = distance.reshape(L * L)
    dist_cutoff = flat_distance.sort()[0][int(L * L * density_t)]
    for i in range(L):
        densities[i] = (distance[i] < dist_cutoff).sum() - 1
    return densities


def pair_feature(p, Q):
    return torch.cat((p.unsqueeze(0).expand_as(Q), Q), dim=1)


def vis_rank(clean_rank, image_list):
    from utils.project_kits import copy_to
    for rank, i in enumerate(clean_rank):
        source_image_path = image_list[i]
        target_image_path = f'pretrained/vis_rank/{rank}.jpg'
        copy_to(source_image_path, target_image_path)
    return


if __name__ == '__main__':
    main()
