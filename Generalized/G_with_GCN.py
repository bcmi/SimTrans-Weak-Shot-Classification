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
from GCN_reg.graph_modules import GCN, CosMimicking
from collections import Counter
from torchvision import models


def add_pretrain(parser):
    parser.add_argument('--exp_type', default='Generalized', type=str)

    parser.add_argument('--weight_type', default='avg1', type=str)
    parser.add_argument('--alpha', default=0, type=float)
    parser.add_argument('--epoch_fraction', default=1, type=float)

    parser.add_argument('--mu', default=0, type=float)
    parser.add_argument('--eye', default=1, type=float)
    parser.add_argument('--class_weight', type=int, default=-1)

    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--similarity_net', type=str, default='')
    parser.add_argument('--resume', type=str, default='none')
    parser.add_argument('--reg', default='GCN', type=str)
    return parser


class Net(nn.Module):
    def __init__(self, resume, class_num):
        super(Net, self).__init__()
        if resume == 'none':
            log('use image net pretrained')
            self.backbone = models.resnet50(pretrained=True)
        else:
            log('load from:')
            log(resume)
            self.backbone = torch.load(resume).module.backbone

        self.class_num = class_num
        self.fc = nn.Linear(2048, class_num)

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
        x = self.fc(x)
        return x, feat


def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_pretrain(parser)
    args = parser.parse_args()

    args.exp_name += f'{args.exp_type}{args.class_weight}_withGCN_{args.reg}_EF{args.epoch_fraction}_a{args.alpha}_{args.weight_type}' \
                     f'_mu{args.mu}_eye{args.eye}' \
                     f'_W{os.path.basename(args.load_dir)}_{os.path.basename(args.data_path)}' \
                     f'_lr{args.lr}_b{args.batch_size}_{args.lr_interval}_{args.lr_decay}' \
                     f'_{datetime.datetime.now().strftime("%m%d%H%M")}'

    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)
    set_seeds(args.seed)

    data_helper = get_data_helper(args)

    if args.exp_type == 'clean':
        train_loader = data_helper.get_all_clean_loader()
    else:
        train_loader = data_helper.get_generalized_loader()

    test_loader = data_helper.get_all_test_loader()

    model = nn.DataParallel(Net(args.resume, class_num=len(test_loader.dataset.categories))).cuda()

    weight_dict = get_weight_dict(args, train_loader.dataset, dataname=data_helper.name)

    train_acc, test_acc = [], []

    # test_meter = evaluation(args, -1, model, test_loader)
    # log(f'Resume from {test_meter}')
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

    if args.class_weight == 1:
        class_weight = torch.FloatTensor(list(Counter([i[1] for i in data_loader.dataset.image_list]).values())).cuda()
        class_weight /= class_weight.mean()
        if epoch < 1:
            log('class weight:')
            log(class_weight)
        cls_criterion = nn.DataParallel(nn.CrossEntropyLoss(class_weight, reduction='none')).cuda()
    elif args.class_weight == -1:
        class_weight = torch.FloatTensor(list(Counter([i[1] for i in data_loader.dataset.image_list]).values())).cuda()
        class_weight = class_weight.mean() / class_weight
        if epoch < 1:
            log('class weight:')
            log(class_weight)
        cls_criterion = nn.DataParallel(nn.CrossEntropyLoss(class_weight, reduction='none')).cuda()
    else:
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

        if args.alpha == 0:
            GCN_loss = cls_loss
        else:
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
            w = torch.tensor(1.)
        ws.append(w)

    return torch.stack(ws)


def get_weight_dict(args, dataset, dataname):
    weight_dict = {}
    for cname in dataset.categories:
        path = f'{args.load_dir}/{cname}'
        if not os.path.exists(path + '_name.pth'):
            continue
        names = torch.load(f'{path}_name.pth')
        similarity_matrix = torch.load(f'{path}_matrix.pth')
        if 'avg' in args.weight_type:
            weights = get_avg(similarity_matrix, args.weight_type)
        elif args.weight_type == 'none':
            weights = similarity_matrix.new_ones(len(similarity_matrix))
        else:
            NotImplementedError

        weights /= weights.mean()

        for name, weight in zip(names, weights):
            weight_dict[name.replace('\\', os.sep)] = weight
    return weight_dict


def get_avg(similarity_matrix, weight_type):
    k = int(float(weight_type[3:]) * len(similarity_matrix))
    return similarity_matrix.topk(k)[0].mean(1)


if __name__ == '__main__':
    main()
