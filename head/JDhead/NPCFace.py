"""
@author: Hand Du, Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class NPCFace(Module):
    """Implementation for "NPCFace: A Negative-Positive Cooperation
       Supervision for Training Large-scale Face Recognition"
    """
    def __init__(self, feat_dim=512, num_class=86876, margin=0.5, scale=64):
        super(NPCFace, self).__init__()
        self.kernel = Parameter(torch.Tensor(feat_dim, num_class))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.m0 = 0.40
        self.m1 = 0.20
        self.t = 1.10
        self.a = 0.20
        self.cos_m0 = math.cos(self.m0)
        self.sin_m0 = math.sin(self.m0)
        self.num_class = num_class

    def forward(self, x, label): 
        kernel_norm = F.normalize(self.kernel, dim=0)
        x = F.normalize(x)
        cos_theta = torch.mm(x, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1) 
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1) 
        sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
        cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m
        with torch.no_grad():
            hard_mask = (cos_theta > cos_theta_m).type(torch.FloatTensor).cuda()
            hard_mask.scatter_(1, label.data.view(-1, 1), 0)
            hard_cos = torch.where(hard_mask > 0, cos_theta, torch.zeros_like(cos_theta))
            sum_hard_cos = torch.sum(hard_cos,dim=1).view(-1, 1)
            sum_hard_mask = torch.sum(hard_mask, dim=1).view(-1,1) 
            sum_hard_mask = sum_hard_mask.clamp(1, self.num_class)  
            avg_hard_cos = sum_hard_cos / sum_hard_mask 
            newm = self.m0 + self.m1 * avg_hard_cos
            cos_newm = torch.cos(newm)
            sin_newm = torch.sin(newm)  
        final_gt = torch.where(gt > 0, gt * cos_newm - sin_theta * sin_newm , gt)
        cos_theta = torch.where(cos_theta > cos_theta_m, self.t * cos_theta + self.a , cos_theta)
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.scale        
        return cos_theta
