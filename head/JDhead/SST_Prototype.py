""" 
@author: Hang Du, Jun Wang 
@date: 20201020
@contact: jun21wangustc@gmail.com   
""" 

import torch
from torch.nn import Module
import math
import random
import torch.nn.functional as F

class SST_Prototype(Module):
    """Implementation for "Semi-Siamese Training for Shallow Face Learning".
    """
    def __init__(self, feat_dim=512, queue_size=16384, scale=30.0, loss_type='softmax', margin=0.0):
        super(SST_Prototype, self).__init__()
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.scale = scale
        self.margin = margin
        self.loss_type = loss_type
        # initialize the prototype queue
        self.register_buffer('queue', torch.rand(feat_dim,queue_size).uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5))
        self.queue = F.normalize(self.queue, p=2, dim=0) # normalize the initial queue.
        self.index = 0
        self.label_list = [-1] * queue_size

    def add_margin(self, cos_theta, label, batch_size):
        cos_theta = cos_theta.clamp(-1, 1) 
        # additive cosine margin
        if self.loss_type == 'am_softmax':
            cos_theta_m = cos_theta[torch.arange(0, batch_size), label].view(-1, 1) - self.margin
            cos_theta.scatter_(1, label.data.view(-1, 1), cos_theta_m)
        # additive angurlar margin
        elif self.loss_type == 'arc_softmax':
            gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * math.cos(self.margin) - sin_theta * math.sin(self.margin) 
            cos_theta.scatter_(1, label.data.view(-1, 1), cos_theta_m)
        return cos_theta

    def compute_theta(self, p, g, label, batch_size):
        queue = self.queue.clone()
        queue[:,self.index:self.index+batch_size] = g.transpose(0,1)
        cos_theta = torch.mm(p, queue.detach())
        cos_theta = self.add_margin(cos_theta, label,batch_size)
        return cos_theta

    def update_queue(self, g, cur_ids, batch_size):
        with torch.no_grad():
            self.queue[:,self.index:self.index+batch_size] = g.transpose(0,1)
            for image_id in range(batch_size):
                self.label_list[self.index + image_id] = cur_ids[image_id].item()
            self.index = (self.index + batch_size) % self.queue_size

    def get_id_set(self):
        id_set = set()
        for label in self.label_list:
            if label != -1:
                id_set.add(label)
        return id_set

    def forward(self, p1, g2, p2, g1, cur_ids):
        p1 = F.normalize(p1)
        g2 = F.normalize(g2)
        p2 = F.normalize(p2)
        g1 = F.normalize(g1)
        batch_size = p1.shape[0]
        label = (torch.LongTensor([range(batch_size)]) + self.index)
        label = label.squeeze().cuda()
        g1 = g1.detach()
        g2 = g2.detach()
        output1 = self.compute_theta(p1, g2, label, batch_size)
        output2 = self.compute_theta(p2, g1, label, batch_size)
        output1 *= self.scale
        output2 *= self.scale
        if random.random() > 0.5:
            self.update_queue(g1, cur_ids, batch_size)
        else:
            self.update_queue(g2, cur_ids, batch_size) 
        id_set = self.get_id_set()
        return output1, output2, label, id_set
