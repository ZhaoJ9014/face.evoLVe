""" 
@author: Hang Du, Jun Wang
@date: 20201128
@contact: jun21wangustc@gmail.com
"""
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

class Adam_Softmax(Module):
    """Implementation for "AdaptiveFace: Adaptive Margin and Sampling for Face Recognition".
    """
    def __init__(self, feat_dim, num_class, scale=30.0, lamda=70.0):
        super(Adam_Softmax, self).__init__()
        self.num_class = num_class
        self.scale = scale
        self.lamda = lamda
        self.kernel = Parameter(torch.Tensor(feat_dim, num_class))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.adam = Parameter(torch.Tensor(1, num_class))
        self.adam.data.uniform_(0.3,0.4) 
    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.kernel, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1,1)
        # margin in [0,1] for cosface.
        self.adam.data.clamp_(0,1)

        margin = self.adam[:, labels].view(-1, 1)
        cos_theta_m = cos_theta - margin
        index = torch.zeros_like(cos_theta)
        index.scatter_(1,labels.data.view(-1,1),1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale

        #ensure the loss > 0
        Lm = -1* torch.sum(self.adam, dim=1)/self.num_class + 1
        return output, self.lamda*Lm
