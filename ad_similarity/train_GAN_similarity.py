import sys
import warnings

sys.path.append(".")
warnings.filterwarnings("ignore")

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


def add_similarity_train(parser):
    parser.add_argument('--batch_class_num', default=20, type=int)
    parser.add_argument('--batch_class_num_B', default=1, type=int)
    parser.add_argument('--target_domain', default='novel_train', type=str)

    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--domain_lr', default=0.01, type=float)

    parser.add_argument('--head_type', default='2', type=str)
    parser.add_argument('--similarity_weighted', default=0, type=int)

    parser.add_argument('--domain_bn', default=1, type=int)

    parser.add_argument('--source_set', default='clean_base', type=str)
    parser.add_argument('--target_set', default='noisy_novel', type=str)

    parser.add_argument('--noisy_frac', default=0, type=float)

    parser.add_argument('--similarity_pretrained', type=str, default='pretrained/CUB/step0/pretrained_84.5.pth')
    return parser


def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_similarity_train(parser)
    args = parser.parse_args()

    args.exp_name += f'GANSimilarity_{args.exp_name}_t{args.target_domain}_beta{args.beta}_{args.source_set}2{args.target_set}_{os.path.basename(args.data_path)}' \
                     f'_lr{args.lr}_b{args.batch_size}_h{args.head_type}' \
                     f'_bnA{args.batch_class_num}_bnB{args.batch_class_num_B}_sw{args.similarity_weighted}_nf{args.noisy_frac}' \
                     f'_{datetime.datetime.now().strftime("%m%d%H%M")}'

    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)
    set_seeds(args.seed)

    data_helper = get_data_helper(args)
    log(data_helper)

    if args.source_set == 'clean_base':
        A_loader = data_helper.get_clean_base_loader()
    elif args.source_set == 'noisy_base':
        A_loader = data_helper.get_noisy_base_loader()
    elif args.source_set == 'noisy_novel':
        A_loader = data_helper.get_noisy_novel_loader()
    elif args.source_set == 'clean_novel':
        A_loader = data_helper.get_clean_novel_loader()
    else:
        raise NotImplementedError

    if args.target_set == 'noisy_novel':
        base_test_loader = data_helper.get_base_test_loader()
        novel_test_loader = data_helper.get_novel_test_loader()
    else:
        raise NotImplementedError

    if args.target_domain == 'novel_train':
        B_loader = data_helper.get_noisy_novel_loader()
    elif args.target_domain == 'novel_test':
        B_loader = novel_test_loader
    else:
        raise NotImplementedError

    base_similarity_test_loader = get_similarity_test_loader2(args, base_test_loader)
    novel_similarity_test_loader = get_similarity_test_loader2(args, novel_test_loader)

    simnet = GANSimilarityNet(args).cuda()
    domain_classifier = nn.DataParallel(DomainClassifier(args, simnet.diff_dim)).cuda()

    train_acc, test_acc, val_acc, val0_acc, domain = [], [], [], [], []

    for epoch in range(args.num_epoch):
        train_meter, domain_meter, cls_loss_avg = train(args, epoch, simnet, domain_classifier, A_loader, B_loader)

        val_meter = evaluation(args, epoch, simnet, base_similarity_test_loader)
        test_meter = evaluation(args, epoch, simnet, novel_similarity_test_loader)

        log(f'\t\t\tEpoch {epoch:3}: '
            f'Test {test_meter.get_main():.3%}; '
            f'Val {val_meter.get_main():.3%}; '
            f'Domain {domain_meter.get_main():.3%}; '
            f'Train {train_meter.get_main():.3%}.')
        log(f'\t\t\tCLS: {cls_loss_avg}.')

        log('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
            '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        log('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'
            '||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n')

        test_acc.append(test_meter.get_main())
        train_acc.append(train_meter.get_main())
        val_acc.append(val_meter.get_main())
        domain.append(domain_meter.get_main())

        saver.save_model_if_best(test_acc, simnet, f'saves/{args.exp_name}/simnet_{args.exp_name}_best.pth',
                                 printf=log, show_acc=False)
        saver.save_model_if_best(test_acc, domain_classifier,
                                 f'saves/{args.exp_name}/domain_classifier_{args.exp_name}_best.pth',
                                 printf=log, show_acc=False)

        if (epoch + 1) % 25 == 0:
            torch.save(simnet, f'saves/{args.exp_name}/simnet_{epoch}_{args.exp_name}_{test_acc[-1]:.3f}.pth')

        if (epoch + 1) % args.report_interval == 0:
            log(f'\n\n##################\n\tBest: {np.max(test_acc):.3%}\n##################\n\n')

            vis_acc([test_acc, train_acc, val_acc, domain],
                    ['Test F1', 'Train F1', 'Val F1', 'Domain'],
                    f'saves/{args.exp_name}/vis_{args.exp_name}_acc{max(test_acc) * 100:.2f}_e{epoch}.jpg')

    return


def train(args, epoch, simnet, domain_classifier, A_loader, B_loader):
    A_pairs_data_loader = get_train_loader_(args, A_loader, B_loader, args.batch_class_num)
    B_pairs_data_loader = get_train_loader_(args, B_loader, B_loader, args.batch_class_num_B)

    lr = args.lr * args.lr_decay ** (epoch // args.lr_interval)
    domain_lr = args.domain_lr * args.lr_decay ** (epoch // args.lr_interval)

    sim_optimizer = optim.SGD(simnet.parameters(), lr=lr, momentum=0.9, weight_decay=args.wd)
    domain_optimizer = optim.SGD(domain_classifier.parameters(), lr=domain_lr, momentum=0.9, weight_decay=args.wd)

    domain_criterion = nn.CrossEntropyLoss().cuda()

    if args.similarity_weighted < 0:
        criterion = nn.CrossEntropyLoss(torch.tensor([1, args.batch_class_num]).float()).cuda()
    elif args.similarity_weighted == 0:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss(torch.tensor([1, args.similarity_weighted]).float()).cuda()

    main_meter = MetrixMeter(['Dissimilarity', 'Similarity'], default_metric='f1score')
    domain_meter = MetrixMeter(['A', 'B'])

    cls_loss_avg = AverageMeter('Training cls loss')

    simnet.train()
    for batch_i, (A_data, B_data) in tqdm(enumerate(zip(A_pairs_data_loader, B_pairs_data_loader))):
        A_images, A_categories, A_names = A_data
        B_images, B_categories, B_names = B_data
        L = len(A_names)

        if len(A_names) != len(B_names):
            print(f'\nIter{batch_i} continued\n')
            continue

        # -----------------
        #  Train Similarity Net
        # -----------------
        # simnet.similarity_head.module.bn
        A_targets = make_similarities(A_categories[0].cuda())
        B_targets = make_similarities(B_categories[0].cuda())
        domain_target = torch.cat((A_targets.new_zeros(len(A_targets)),
                                   B_targets.new_ones(len(B_targets)))).long().clone().detach()

        sim_optimizer.zero_grad()

        AB_feat = simnet.backbone(torch.cat((A_images[0], B_images[0])).cuda())
        A_feat, B_feat = AB_feat.chunk(2)

        A_pairs = pair_enumeration(A_feat)
        B_pairs = pair_enumeration(B_feat)

        AB_similarities, AB_diff_feat = simnet.similarity_head(torch.cat((A_pairs, B_pairs)))
        A_predictions, B_predictions = AB_similarities.chunk(2)
        A_diff_feat, B_diff_feat = AB_diff_feat.chunk(2)

        domain_pred = domain_classifier(torch.cat((A_diff_feat, B_diff_feat)))
        domain_loss = domain_criterion(domain_pred, domain_target)

        cls_loss = criterion(A_predictions, A_targets)

        total_loss = cls_loss.mean() - args.beta * domain_loss

        total_loss.backward()

        sim_optimizer.step()
        sim_optimizer.zero_grad()

        main_meter.update(A_predictions, A_targets)
        cls_loss_avg.update(cls_loss.mean().item())

        # ---------------------
        #  Train Discriminator
        # ---------------------

        domain_optimizer.zero_grad()
        domain_pred = domain_classifier(torch.cat((A_diff_feat, B_diff_feat)).clone().detach())
        domain_loss = domain_criterion(domain_pred, domain_target)
        domain_loss.backward()
        domain_optimizer.step()
        domain_optimizer.zero_grad()

        domain_meter.update(domain_pred, domain_target)

    log('Similarity')
    log(main_meter.report())
    log('Domain')
    log(domain_meter.report())

    return main_meter, domain_meter, cls_loss_avg


def evaluation(args, epoch, model, data_loader):
    meter = MetrixMeter(['Dissimilarity', 'Similarity'], default_metric='f1score')
    model.eval()

    with torch.no_grad():
        for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
            predictions = model(images.cuda())
            targets = make_similarities(categories)
            meter.update(predictions, targets)
    log(meter.report())
    return meter


if __name__ == '__main__':
    main()
