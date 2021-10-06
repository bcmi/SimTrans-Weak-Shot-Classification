import sys
import warnings

sys.path.append(".")
warnings.filterwarnings("ignore")
import os
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
from utils.project_kits import init_log, log, set_seeds, occupy, early_stop
from utils.vis import vis_acc
from collections import Counter
from Generalized.G_modules import *


def add_naive_train(parser):
    parser.add_argument('--exp_type', default='Generalized', type=str)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--class_weight', type=int, default=0)
    parser.add_argument('--epoch_iter', type=int, default=1000)
    parser.add_argument('--sampler', type=int, default=1)

    return parser


def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_naive_train(parser)
    args = parser.parse_args()

    args.exp_name += f'{args.exp_type}_W{args.class_weight}S{args.sampler}_{os.path.basename(args.data_path)}_lr{args.lr}_b{args.batch_size}_wd{args.wd}_' \
                     f'{datetime.datetime.now().strftime("%m%d%H%M")}'

    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)
    set_seeds(args.seed)

    data_helper = get_data_helper(args)
    log(data_helper)

    if args.exp_type == 'clean':
        train_loader = data_helper.get_all_clean_loader()
    else:
        train_loader = data_helper.get_generalized_loader()

    test_loader = data_helper.get_all_test_loader()

    log(len(test_loader.dataset.categories))
    log(test_loader.dataset.categories)
    model = nn.DataParallel(mhelper.get_model(args, train_loader.dataset.num_classes(), resume=args.resume)).cuda()

    train_acc, test_acc = [], []
    # test_loader.dataset.int2category[21], train_loader.dataset.int2category[21]
    for epoch in range(args.num_epoch):
        train_meter, train_log = train(args, epoch, model, train_loader)
        test_meter = evaluation(args, epoch, model, test_loader)

        log(f'\t\t\tEpoch {epoch:3}: Test {test_meter}; Train {train_meter}.')
        log(f'\t\t\tCLS: [{np.mean(train_log[0]):4.6f}]')

        test_acc.append(test_meter.acc())
        train_acc.append(train_meter.acc())
        saver.save_model_if_best(test_acc, model, f'saves/{args.exp_name}/{args.exp_name}_best.pth',
                                 printf=log, show_acc=False)

        early_stop(test_acc, args.stop_th, log)

        if (epoch + 1) % args.report_interval == 0:
            log(f'\n\n##################\n\tBest: {np.max(test_acc):.3%}\n##################\n\n')

            vis_acc([test_acc, train_acc],
                    ['Test Acc', 'Train Acc'],
                    f'saves/{args.exp_name}/acc_e{epoch}_{max(test_acc) * 100:.2f}.jpg')
            log(test_meter.report(hit=False))
    return


def train(args, epoch, model, data_loader):
    meter = MetrixMeter(data_loader.dataset.categories)

    if args.sampler == 1:
        data_loader = get_train_loader(args, data_loader)

    lr = args.lr * args.lr_decay ** (epoch // args.lr_interval)

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr, 'wd': args.wd}])
    else:
        raise NotImplementedError

    if args.class_weight == 1:
        class_weight = torch.FloatTensor(list(Counter([i[1] for i in data_loader.dataset.image_list]).values())).cuda()
        class_weight /= class_weight.mean()
        cls_criterion = nn.DataParallel(nn.CrossEntropyLoss(class_weight)).cuda()
    elif args.class_weight == -1:
        class_weight = torch.FloatTensor(list(Counter([i[1] for i in data_loader.dataset.image_list]).values())).cuda()
        class_weight = class_weight.mean() / class_weight
        cls_criterion = nn.DataParallel(nn.CrossEntropyLoss(class_weight)).cuda()
    else:
        cls_criterion = nn.DataParallel(nn.CrossEntropyLoss()).cuda()

    model.train()

    losslog = [[], [], [], [], []]
    for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        predictions = model(images.cuda())
        cls_loss = cls_criterion(predictions, categories)
        total_loss = cls_loss.mean()

        total_loss.backward()
        optimizer.step()

        meter.update(predictions, categories)
        losslog[0].append(cls_loss.mean().item())

    return meter, losslog


def evaluation(args, epoch, model, data_loader):
    meter = MetrixMeter(data_loader.dataset.categories)
    model.eval()

    with torch.no_grad():
        for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
            predictions = model(images.cuda())
            meter.update(predictions, categories)

    return meter


if __name__ == '__main__':
    main()
