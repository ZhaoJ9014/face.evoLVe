"""
@author: Jun Wang
@date: 20201128
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class AdaCos(nn.Module):
    """Implementation for "Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations".
    """
    def __init__(self, feat_dim, num_classes):
        super(AdaCos, self).__init__()
        self.scale = math.sqrt(2) * math.log(num_classes - 1)
        self.W = Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.W)
    def forward(self, feats, labels):
        # normalize weights
        W = F.normalize(self.W)
        # normalize feats
        feats = F.normalize(feats)
        # dot product
        logits = F.linear(feats, W)
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.scale * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / feats.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.scale = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.scale * logits
        return output
