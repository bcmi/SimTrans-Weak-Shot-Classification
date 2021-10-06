import sys
import warnings

sys.path.append(".")
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import utils.saver as saver
from data.factory import get_data_helper
import models.helper as mhelper

from utils.meters import MetrixMeter
from utils.parser import get_base_parser, add_base_train
from utils.project_kits import init_log, log, set_seeds, occupy, copy_to, early_stop
from utils.vis import vis_acc
import os
from GCN_reg.graph_modules import *


def add_pretrain(parser):
    parser.add_argument('--weight_type', default='avg1', type=str)
    parser.add_argument('--alpha', default=0, type=float)
    parser.add_argument('--epoch_fraction', default=1, type=float)

    parser.add_argument('--mu', default=0, type=float)
    parser.add_argument('--eye', default=1, type=float)

    parser.add_argument('--load_dir', type=str, default='pretrained/similarity/clean_base')
    parser.add_argument('--similarity_net', type=str,
                        default='pretrained/similarity_CUB_lr0.01_b100_h2_clean_base_04271337_81.5.pth')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reg', default='GCN', type=str)

    return parser


def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_pretrain(parser)
    args = parser.parse_args()

    args.exp_name += f'with{args.reg}_EF{args.epoch_fraction}_a{args.alpha}_{args.weight_type}' \
                     f'_W{os.path.basename(args.load_dir)}_{os.path.basename(args.data_path)}' \
                     f'_lr{args.lr}_b{args.batch_size}_{args.lr_interval}_{args.lr_decay}' \
                     f'_{datetime.datetime.now().strftime("%m%d%H%M")}'

    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)
    set_seeds(args.seed)

    data_helper = get_data_helper(args)

    train_loader = data_helper.get_noisy_novel_loader()
    test_loader = data_helper.get_novel_test_loader()

    model = nn.DataParallel(Net(mhelper.get_model(args, train_loader.dataset.num_classes(), args.resume))).cuda()

    weight_dict = get_weight_dict(args, train_loader.dataset, dataname=data_helper.name)

    train_acc, test_acc = [], []

    test_meter = evaluation(args, -1, model, test_loader)
    log(f'Resume from {test_meter}')
    for epoch in range(args.num_epoch):
        train_meter, train_log = train(args, epoch, model, train_loader, weight_dict)
        test_meter = evaluation(args, epoch, model, test_loader)

        log(f'\t\t\tEpoch {epoch:3}: Test {test_meter}; Train {train_meter}.')
        log(f'\t\t\tCLS: [{np.mean(train_log[0]):4.6f}], GCN: [{np.mean(train_log[1]):4.10f}]')

        test_acc.append(test_meter.acc())
        train_acc.append(train_meter.acc())
        saver.save_model_if_best(test_acc, model, f'saves/{args.exp_name}/{args.exp_name}_best.pth',
                                 printf=log, show_acc=False)

        if (epoch + 1) % args.report_interval == 0:
            log(f'\n\n##################\n\tBest: {np.max(test_acc):.3%}\n##################\n\n')

            vis_acc([test_acc, train_acc],
                    ['Test Acc', 'Train Acc'],
                    f'saves/{args.exp_name}/acc_e{epoch}_{max(test_acc) * 100:.2f}.jpg')

        early_stop(test_acc, args.stop_th)

    return


def train(args, epoch, model, data_loader, weight_dict):
    meter = MetrixMeter(data_loader.dataset.categories)
    lr = args.lr * args.lr_decay ** (epoch // args.lr_interval)

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr, 'wd': args.wd}])
    else:
        raise NotImplementedError

    if args.reg == 'GCN':
        reg_loss = GCN
    elif args.reg == 'COS':
        reg_loss = CosMimicking
    else:
        raise NotImplementedError

    cls_criterion = nn.DataParallel(nn.CrossEntropyLoss(reduction='none')).cuda()
    model.train()

    simnet = torch.load(args.similarity_net)
    simnet.reset_gpu()
    simnet.eval()

    losslog = [[], [], [], [], []]
    stop_idx = int(len(data_loader) * args.epoch_fraction)

    for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
        if batch_i > stop_idx:
            continue

        optimizer.zero_grad()

        predictions, feats = model(images.cuda())
        cls_loss = cls_criterion(predictions, categories)
        weights = get_weights(args, categories, im_names, weight_dict)

        GCN_loss = reg_loss(args, images, feats, simnet, categories)

        total_loss = (cls_loss * weights.type_as(cls_loss)).mean() + args.alpha * GCN_loss.mean()

        total_loss.backward()
        optimizer.step()

        meter.update(predictions, categories)
        losslog[0].append(cls_loss.mean().item())
        losslog[1].append(GCN_loss.mean().item())

    return meter, losslog


def evaluation(args, epoch, model, data_loader):
    meter = MetrixMeter(data_loader.dataset.categories)

    model.eval()

    with torch.no_grad():
        for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
            predictions, _ = model(images.cuda())
            meter.update(predictions, categories)

    return meter


def get_weights(args, categories, im_names, weight_dict):
    ws = []
    for k in im_names:
        if weight_dict.__contains__(k):
            w = weight_dict[k]
        else:
            log(f'key not found: {k}')
            w = torch.tensor(1.)
        ws.append(w)

    return torch.stack(ws)


def get_weight_dict(args, dataset, dataname):
    weight_dict = {}
    for cname in dataset.categories:
        path = f'{args.load_dir}/{cname}'
        names = torch.load(f'{path}_name.pth')
        similarity_matrix = torch.load(f'{path}_matrix.pth')
        if 'den' in args.weight_type:
            weights = get_density(1 - similarity_matrix, float(args.weight_type[3:]))
        elif 'avg' in args.weight_type:
            weights = get_avg(similarity_matrix, args.weight_type)
        elif 'order' in args.weight_type:
            weights = get_order(similarity_matrix, args.weight_type)
        elif args.weight_type == 'none':
            weights = similarity_matrix.new_ones(len(similarity_matrix))
        elif 'kmeans' in args.weight_type:
            weights = kmeans(similarity_matrix, args.weight_type, names)
            weights = similarity_matrix.new(weights)
        else:
            NotImplementedError

        weights /= weights.mean()

        for name, weight in zip(names, weights):
            weight_dict[name.replace('\\', os.sep)] = weight
    return weight_dict


def kmeans(smatrix, weight_type, names):
    def vis_cluster(Gs, names):
        sizes = [len(G) for G in Gs]
        args = np.argsort(sizes)[::-1]

        for i, arg in enumerate(args):
            G = Gs[arg]
            for idx in G:
                source_path = names[idx]
                target_path = f'pretrained/tmp/{i}/{idx}.jpg'
                copy_to(source_path, target_path)
        return

    from similarity.graph_cluster import K_cluster
    K = int(weight_type.split('-')[1])
    Gs = K_cluster(smatrix, K)
    # vis_cluster(Gs, names)
    sizes = torch.tensor([len(G) for G in Gs]).float()
    G_weights = sizes / sizes.mean()
    weights = smatrix.new_zeros(len(smatrix))

    for i, G in enumerate(Gs):
        for g in G:
            weights[g] = G_weights[i]

    sm = [smatrix[G, :][:, G].mean() for G in Gs]
    return


def get_order(similarity_matrix, weight_type):
    os = [int(o) for o in weight_type.split('-')[1:]]

    def compute_order(matrix, order):
        if order == 1:
            return matrix
        else:
            last = compute_order(matrix, order - 1)
            last_norm = (last ** 2).sum(1, keepdim=True).sqrt()

            A = torch.matmul(last, last.transpose(1, 0)) / torch.matmul(last_norm, last_norm.transpose(1, 0))

            return A

    Ss = [compute_order(similarity_matrix, o) for o in os]
    return torch.stack(Ss).mean(0).mean(1)


def get_avg(similarity_matrix, weight_type):
    k = int(float(weight_type[3:]) * len(similarity_matrix))
    return similarity_matrix.topk(k)[0].mean(1)


def get_density(distance, density_t=0.6):
    log(f'density threshold: {density_t}')
    L = len(distance)
    densities = distance.new_zeros(L)
    flat_distance = distance.reshape(L * L)
    dist_cutoff = flat_distance.sort()[0][int(L * L * density_t)]
    for i in range(L):
        densities[i] = (distance[i] < dist_cutoff).sum() - 1
    return densities


def vis_weights(weights, names, category_name):
    from utils.project_kits import copy_to
    for rank, i in enumerate(weights.sort()[1]):
        source_image_path = names[i]
        target_image_path = f'pretrained/vis_rank/{category_name}/{rank}_{weights[i]}.jpg'
        copy_to(source_image_path, target_image_path)
    return


if __name__ == '__main__':
    main()
