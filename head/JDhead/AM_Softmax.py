"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class AM_Softmax(Module):
    """Implementation for "Additive Margin Softmax for Face Verification"
    """
    def __init__(self, feat_dim, num_class, margin=0.35, scale=32):
        super(AM_Softmax, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.scale = scale
    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta_m = cos_theta - self.margin
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output
