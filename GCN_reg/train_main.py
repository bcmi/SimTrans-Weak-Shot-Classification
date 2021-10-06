import sys
import warnings

sys.path.append(".")
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

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


def add_pretrain(parser):
    parser.add_argument('--weight_type', default='savg1', type=str)
    parser.add_argument('--weight_norm', default='one_mean', type=str)
    parser.add_argument('--epoch_fraction', default=1, type=float)
    parser.add_argument('--resume', default=None, type=str)

    parser.add_argument('--load_dir', type=str, default='pretrained/CUB/step2/h4f2_b0.1_86.5')
    return parser


def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_pretrain(parser)
    args = parser.parse_args()

    args.exp_name += f'main_EF{args.epoch_fraction}_{args.weight_norm}_{args.weight_type}_W{os.path.basename(args.load_dir)}_{os.path.basename(args.data_path)}' \
                     f'_lr{args.lr}_b{args.batch_size}_{args.lr_interval}_{args.lr_decay}_' \
                     f'{datetime.datetime.now().strftime("%m%d%H%M")}'

    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)
    set_seeds(args.seed)

    data_helper = get_data_helper(args)

    train_loader = data_helper.get_noisy_novel_loader()
    test_loader = data_helper.get_novel_test_loader()

    model = nn.DataParallel(mhelper.get_model(args, train_loader.dataset.num_classes(), args.resume)).cuda()

    weight_dict = get_weight_dict(args, train_loader.dataset, dataname=data_helper.name)

    train_acc, test_acc = [], []

    for epoch in range(args.num_epoch):
        train_meter, train_log = train(args, epoch, model, train_loader, weight_dict)
        test_meter = evaluation(args, epoch, model, test_loader)

        log(f'\t\t\tEpoch {epoch:3}: Test {test_meter}; Train {train_meter}.')
        log(f'\t\t\tCLS: [{np.mean(train_log[0]):4.6f}]')

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

    cls_criterion = nn.DataParallel(nn.CrossEntropyLoss(reduction='none')).cuda()
    model.train()
    losslog = [[], [], [], [], []]

    stop_idx = int(len(data_loader) * args.epoch_fraction)

    for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
        if batch_i > stop_idx:
            continue

        optimizer.zero_grad()

        predictions = model(images.cuda())
        cls_loss = cls_criterion(predictions, categories)

        weights = get_weights(args, categories, im_names, weight_dict)
        total_loss = (cls_loss * weights.type_as(cls_loss)).mean()

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
    means = []
    for cname in dataset.categories:
        path = f'{args.load_dir}/{cname}'
        saved_names = torch.load(f'{path}_name.pth')
        similarity_matrix = torch.load(f'{path}_matrix.pth')
        names = saved_names

        if args.weight_type.startswith('avg'):
            weights = get_avg(similarity_matrix, args.weight_type)
        elif args.weight_type.startswith('savg'):
            weights = get_savg(similarity_matrix, args.weight_type)
        elif args.weight_type == 'none':
            weights = similarity_matrix.new_ones(len(similarity_matrix))
        else:
            raise NotImplementedError

        if 'one_mean' in args.weight_norm:
            weights /= weights.mean()
        elif 'max' in args.weight_norm:
            weights /= weights.max()
            max, mu = [float(t) for t in args.weight_norm.split('_')[1:]]
            weights = relax(weights, mu=mu, max=max)
        else:
            raise NotImplementedError

        # names = [(dataset.root_path + '/Dog_web' + n.split('Dog_web')[1]).replace('\\', os.sep).replace('/', os.sep) for
        #          n in saved_names]
        # vis_weights(weights, names, cname)

        for name, weight in zip(names, weights):
            # web_dir = os.path.basename(dataset.root_path) + '_web'
            # p = (dataset.root_path + f'/{web_dir}' + name.split(web_dir)[1]).replace('\\', os.sep).replace('/', os.sep)
            # weight_dict[p] = weight

            weight_dict[name.replace('\\', os.sep)] = weight

        means.append(similarity_matrix.mean())
    return weight_dict


def relax(x, mu=0, max=2):
    if mu == 0:
        return x * max
    else:
        return max / (np.exp(mu) - 1) * (torch.exp(mu * x) - 1)

def get_avg(similarity_matrix, weight_type):
    k = int(float(weight_type[3:]) * len(similarity_matrix))
    return similarity_matrix.topk(k)[0].mean(1)


def get_savg(similarity_matrix, weight_type):
    k = int(float(weight_type[4:]) * len(similarity_matrix))
    return (similarity_matrix.topk(k)[0].mean(1) + similarity_matrix.transpose(1, 0).topk(k)[0].mean(1)) / 2


if __name__ == '__main__':
    main()
