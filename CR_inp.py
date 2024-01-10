import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models


class Resnet152(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet152, self).__init__()
        res_pretrained_features = models.resnet152(pretrained=True)

        self.slice0 = torch.nn.Sequential()
        self.slice1 = torch.nn.Sequential(*list(res_pretrained_features.children())[:-5])
        self.slice2 = torch.nn.Sequential(*list(res_pretrained_features.children())[-5:-4])
        self.slice3 = torch.nn.Sequential(*list(res_pretrained_features.children())[-4:-3])
        self.slice4 = torch.nn.Sequential(*list(res_pretrained_features.children())[-3:-2])

        for x in range(3):
            self.slice0.add_module(str(x), self.slice1[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu0 = self.slice0(X)            # 1th
        h_relu1 = self.slice1(X)            # 11th
        h_relu2 = self.slice2(h_relu1)      # 35th
        h_relu3 = self.slice3(h_relu2)      # 143th
        # h_relu4 = self.slice4(h_relu3)    # 152th
        return [h_relu0, h_relu1, h_relu2]


class ContrastLoss_res(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss_res, self).__init__()
        self.vgg = Resnet152().cuda()
        self.l1 = nn.L1Loss()
        # self.weights = [ 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0, 1.0, 1.0]
        self.ab = ablation

    def forward(self, a, p, n, a_half, n_DC2):
        # a_DC2: 在第二阶段decoder层输出的无雾anchor
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        a_half_vgg, n_DC2_vgg = self.vgg(a_half), self.vgg(n_DC2)
        loss = 0
        d_ap, d_an, d_an_half_dc2= 0, 0, 0
        for i in range(len(a_vgg)):
            a, p, n = a_vgg[i], p_vgg[i], n_vgg[i]
            a_half, n_DC2 = a_half_vgg[i], n_DC2_vgg[i]
            d_ap = self.l1(a, p.detach())       # 正对的L1
            if not self.ab:
                d_an = self.l1(a, n.detach())   # 负对的L1
                d_an_half_dc2 = self.l1(a_half, n_DC2.detach())
                contrastive = d_ap / (d_an + d_an_half_dc2 + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

    # def forward(self, a, p, n):
    #     # a_DC2: 在第二阶段decoder层输出的无雾anchor
    #     a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
    #     loss = 0
    #     d_ap, d_an= 0, 0
    #     for i in range(len(a_vgg)):
    #         a, p, n = a_vgg[i], p_vgg[i], n_vgg[i]
    #         d_ap = self.l1(a, p.detach())       # 正对的L1
    #         if not self.ab:
    #             d_an = self.l1(a, n.detach())   # 负对的L1
    #             contrastive = d_ap / (d_an + 1e-7)
    #         else:
    #             contrastive = d_ap
    #
    #         loss += self.weights[i] * contrastive
    #     return loss
if __name__ == '__main__':
    import torchvision.models as models

    res_pretrained_features = models.resnet152(pretrained=False)
    slice1_list = torch.nn.Sequential()
    slice1 = torch.nn.Sequential(*list(res_pretrained_features.children())[:-5])
    for x in range(3):
        slice1_list.add_module(str(x), slice1[x])

    print(slice1_list)

    # res_pretrained_features = models.vgg19(pretrained=False)
    # print(res_pretrained_features)