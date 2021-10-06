import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, resume):
        super(Net, self).__init__()
        self.backbone = resume

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


def rectify(mu, eye, A, categories):
    def relax(matrix, mu=0):
        if mu == 0:
            return matrix
        else:
            return 1 / (np.exp(mu) - 1) * (torch.exp(mu * matrix) - 1)

    C = ((categories.unsqueeze(0) - categories.unsqueeze(1)) == 0).float().type_as(A)

    A_ = relax(A, mu)
    A__ = A_ * C.clamp(min=eye)
    return A__


def GCN(args, images, feats, simnet, categories):
    with torch.no_grad():
        similarities = simnet(images)
        A = torch.softmax(similarities, dim=1)[:, 1].reshape(images.size(0), -1)

    A = rectify(args.mu, args.eye, A, categories)
    Dist = ((feats.unsqueeze(1) - feats.unsqueeze(0)) ** 2).sum(2)
    L = Dist * A
    return L


def CosMimicking(args, images, feats, simnet, categories):
    with torch.no_grad():
        similarities = simnet(images)
        A = torch.softmax(similarities, dim=1)[:, 1].reshape(images.size(0), -1)

    A = rectify(args.mu, args.eye, A, categories)

    norm = (feats ** 2).sum(1, keepdim=True).sqrt()
    cos_A = torch.matmul(feats, feats.transpose(1, 0)) / torch.matmul(norm, norm.transpose(1, 0))
    L = (cos_A - A) ** 2

    return L
