import torch
from torchvision import models
import torch.nn as nn


def get_model(args, num_classes, resume=None):
    if args.imagenet_pretrained:
        print('use image net pretrained')
        model = eval(f'models.{args.backbone}')(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        print('learn from scratch')
        model = eval(f'models.{args.backbone}')(num_classes=num_classes, pretrained=False)

    if resume is not None:
        saved = torch.load(resume)['state_dict']

        model = eval(f'models.{args.backbone}')(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 972)
        model = nn.DataParallel(model)
        model.load_state_dict(saved)
        model = model.module
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
