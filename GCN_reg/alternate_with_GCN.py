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
from torch.utils.data import DataLoader as Loader

from utils.meters import MetrixMeter
from utils.parser import get_base_parser, add_base_train
from utils.project_kits import init_log, log, set_seeds, occupy, copy_to, early_stop
from utils.vis import vis_acc
import os
from GCN_reg.graph_modules import GCN, CosMimicking


def add_pretrain(parser):
    parser.add_argument('--weight_type', default='avg1', type=str)
    parser.add_argument('--alpha', default=0.1, type=float)

    parser.add_argument('--c0g1', default=1, type=int)

    parser.add_argument('--cls', default=8, type=int)
    parser.add_argument('--gcn', default=128, type=int)

    parser.add_argument('--mu', default=-4, type=float)
    parser.add_argument('--eye', default=1, type=float)

    parser.add_argument('--similarity_net', type=str, default='pretrained/CUB/step1/h4f2_86.5.pth')
    parser.add_argument('--load_dir', type=str, default='pretrained/CUB/step2/h4f2_b0.1_86.5')
    parser.add_argument('--resume', type=str, default='pretrained/CUB/step3/main_90.7.pth')

    parser.add_argument('--reg', default='GCN', type=str)
    return parser


class Net(nn.Module):
    def __init__(self, resume):
        super(Net, self).__init__()
        self.backbone = torch.load(resume).module

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        feat = x
        x = self.backbone.fc(x)
        return x, feat


def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_pretrain(parser)
    args = parser.parse_args()

    args.exp_name += f'Alternate{args.c0g1}_{args.reg}_g{args.gcn}c{args.cls}_a{args.alpha}' \
                     f'_mu{args.mu}_eye{args.eye}' \
                     f'_{args.weight_type}' \
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

    model = nn.DataParallel(Net(args.resume)).cuda()

    weight_dict = get_weight_dict(args, train_loader.dataset, dataname=data_helper.name)

    train_acc, test_acc = [], []

    test_meter = evaluation(args, -1, model, test_loader)
    log(f'Resume from {test_meter}')
    for epoch in range(args.num_epoch):

        if epoch % 2 == args.c0g1:
            train_meter, train_log = train_cls(args, epoch, model, train_loader, weight_dict)
        else:
            train_meter, train_log = train_gcn(args, epoch, model, train_loader, weight_dict)

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


def train_cls(args, epoch, model, data_loader, weight_dict):
    data_loader = Loader(data_loader.dataset, batch_size=args.cls, shuffle=True, num_workers=args.num_workers)
    meter = MetrixMeter(data_loader.dataset.categories)
    lr = args.lr * args.lr_decay ** (epoch // args.lr_interval)

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr, 'wd': args.wd}])
    else:
        raise NotImplementedError
    cls_criterion = nn.DataParallel(nn.CrossEntropyLoss(reduction='none')).cuda()
    model.train()

    simnet = torch.load(args.similarity_net)
    simnet.reset_gpu()
    simnet.eval()

    losslog = [[], [], [], [], []]
    for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        predictions, feats = model(images.cuda())
        cls_loss = cls_criterion(predictions, categories)
        weights = get_weights(args, categories, im_names, weight_dict)

        # GCN_loss = reg_loss(args, images, feats, simnet, categories)
        GCN_loss = cls_loss

        total_loss = (cls_loss * weights.type_as(cls_loss)).mean()

        total_loss.backward()
        optimizer.step()

        meter.update(predictions, categories)
        losslog[0].append(cls_loss.mean().item())
        losslog[1].append(GCN_loss.mean().item())

    return meter, losslog


def train_gcn(args, epoch, model, data_loader, weight_dict):
    data_loader = Loader(data_loader.dataset, batch_size=args.gcn, shuffle=True, num_workers=args.num_workers)

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
    for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
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
            log(f'size of weight dict: {len(weight_dict)}')
            log(f'key not found: {k}')
            print(str(weight_dict)[:100])
            w = torch.tensor(1.)
            # exit(0)
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
        elif args.weight_type == 'none':
            weights = similarity_matrix.new_ones(len(similarity_matrix))
        else:
            NotImplementedError

        weights /= weights.mean()

        for name, weight in zip(names, weights):
            weight_dict[name.replace('\\', os.sep)] = weight

    log(f'size of weight dict: {len(weight_dict)}')
    return weight_dict


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


if __name__ == '__main__':
    main()
