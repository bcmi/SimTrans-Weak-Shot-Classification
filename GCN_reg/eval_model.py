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
import torch.nn.functional as F
from utils.meters import MetrixMeter
from utils.parser import get_base_parser, add_base_train
from utils.project_kits import init_log, log, set_seeds, occupy, copy_to, early_stop
from utils.vis import vis_acc
import os
from GCN_reg.graph_modules import GCN, CosMimicking


def add_pretrain(parser):
    parser.add_argument('--model_path', type=str, default='pretrained/CUB/step3/final_classifier.pth')
    return parser


# class Net(nn.Module):
#     def __init__(self, resume):
#         super(Net, self).__init__()
#         self.backbone = torch.load(resume).module
#
#     def forward(self, x):
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)
#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)
#         x = self.backbone.avgpool(x)
#         x = x.view(x.size(0), -1)
#         feat = x
#         x = self.backbone.fc(x)
#         return x, feat

class Net(nn.Module):
    def __init__(self, args, class_num):
        super(Net, self).__init__()
        self.backbone = mhelper.get_model(args, class_num)

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
    args.exp_name += f'_{datetime.datetime.now().strftime("%m%d%H%M")}'
    init_log(args.exp_name)
    log(args)

    data_helper = get_data_helper(args)
    log(data_helper)
    test_loader = data_helper.get_novel_test_loader()

    if 'params' in args.model_path:
        model = Net(args, test_loader.dataset.num_classes())
        WV = torch.load(args.model_path)
        model.load_state_dict(WV)
    else:
        model = torch.load(args.model_path).module

    model = nn.DataParallel(model).cuda()

    test_meter = evaluation(args, -1, model, test_loader)
    log(f'Eval Result: {test_meter}')
    return


def evaluation(args, epoch, model, data_loader):
    meter = MetrixMeter(data_loader.dataset.categories)
    model.eval()
    with torch.no_grad():
        for batch_i, (images, categories, im_names) in tqdm(enumerate(data_loader)):
            predictions, _ = model(images.cuda())
            meter.update(predictions, categories)

    return meter


if __name__ == '__main__':
    main()
