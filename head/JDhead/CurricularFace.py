"""
@author: Jun Wang
@date: 20201126
@contact: jun21wangustc@gmail.com
"""

# based on
# https://github.com/HuangYG123/CurricularFace/blob/master/head/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class CurricularFace(nn.Module):
    """Implementation for "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition".
    """
    def __init__(self, feat_dim, num_class, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(feat_dim, num_class))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.kernel, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, feats.size(0)), labels].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output
