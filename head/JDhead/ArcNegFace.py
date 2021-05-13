"""
@author: Yaobin Li
@date: 20210219
@contact: cavallyb@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcNegFace(nn.Module):
    """Implement of Towards Flops-constrained Face Recognition (https://arxiv.org/pdf/1909.00632.pdf):
    """
    def __init__(self, feat_dim, num_class, margin=0.5, scale=64):
        super(ArcNegFace, self).__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(num_class, feat_dim))
        self.reset_parameters()
        self.alpha = 1.2
        self.sigma = 2
        self.thresh = math.cos(math.pi-self.margin)
        self.mm = math.sin(math.pi-self.margin) * self.margin
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feats, labels):
        ex = feats / torch.norm(feats, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cos = torch.mm(ex, ew.t())

        a = torch.zeros_like(cos)
        b = torch.zeros_like(cos)
        a_scale = torch.zeros_like(cos)
        c_scale = torch.ones_like(cos)
        t_scale = torch.ones_like(cos)
        for i in range(a.size(0)):
            lb = int(labels[i])
            a_scale[i,lb]=1
            c_scale[i,lb]=0
            if cos[i, lb].item() > self.thresh:
                a[i, lb] = torch.cos(torch.acos(cos[i, lb])+self.margin)
            else:
                a[i, lb] = cos[i, lb]-self.mm
            reweight = self.alpha*torch.exp(-torch.pow(cos[i,]-a[i,lb].item(),2)/self.sigma)
            t_scale[i]*=reweight.detach()
        return self.scale * (a_scale*a+c_scale*(t_scale*cos+t_scale-1))
