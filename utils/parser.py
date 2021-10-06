import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(description='PyTorch Weak-shot Learning')

    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--novel_split', default=-1, type=int)

    parser.add_argument('--data_path', default='../../dataset/CUB', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--occupy', default=0, type=int)
    parser.add_argument('--stop_th', default=10, type=int)
    return parser


def add_base_train(parser):
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)

    parser.add_argument('--lr_interval', default=20, type=int)
    parser.add_argument('--lr_decay', default=0.5, type=float)

    parser.add_argument('--report_interval', default=5, type=int)
    parser.add_argument('--vis_interval', default=5, type=int)
    parser.add_argument('--vis_num', default=20, type=int)
    parser.add_argument('--save_interval', default=5, type=int)

    parser.add_argument('--optim', default='sgd', type=str)

    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--imagenet_pretrained', default=1, type=int)

    return parser
