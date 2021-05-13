"""
@author:Jun Wang
@date: 20201123
@contact: jun21wangustc@gmail.com
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class CircleLoss(Module):
    """Implementation for "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    Note: this is the classification based implementation of circle loss.
    """
    def __init__(self, feat_dim, num_class, margin=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.gamma = gamma

        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1-margin
        self.delta_n = margin

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        index_pos = torch.zeros_like(cos_theta)        
        index_pos.scatter_(1, labels.data.view(-1, 1), 1)
        index_pos = index_pos.byte()
        index_neg = torch.ones_like(cos_theta)        
        index_neg.scatter_(1, labels.data.view(-1, 1), 0)
        index_neg = index_neg.byte()

        alpha_p = torch.clamp_min(self.O_p - cos_theta.detach(), min=0.)
        alpha_n = torch.clamp_min(cos_theta.detach() - self.O_n, min=0.)

        logit_p = alpha_p * (cos_theta - self.delta_p)
        logit_n = alpha_n * (cos_theta - self.delta_n)

        output = cos_theta * 1.0
        output[index_pos] = logit_p[index_pos]
        output[index_neg] = logit_n[index_neg]
        output *= self.gamma
        return output
