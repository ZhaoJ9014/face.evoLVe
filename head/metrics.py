from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


# Support: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
# Support: ['AdaCos','AdaM_Softmax','ArcFace','ArcNegFace','CircleLoss','CurricularFace','MagFace','NPCFace','MV_Softmax','SST_Prototype']
class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        """
    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zero_(self.bias)

    def forward(self, x):
        if self.device_id == None:
            out = F.linear(x, self.weight, self.bias)
        else:
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.50, easy_margin = False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
      
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1) 
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

class SphereFace(nn.Module):
    r"""Implement of SphereFace (https://arxiv.org/pdf/1704.08063.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, device_id, m = 4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cos_theta = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cos_theta = torch.cat((cos_theta, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', m = ' + str(self.m) + ')'


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class Am_softmax(nn.Module):
    r"""Implement of Am_softmax (https://arxiv.org/pdf/1801.05599.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        s: scale of outputs
    """
    def __init__(self, in_features, out_features, device_id, m = 0.35, s = 30.0):
        super(Am_softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.device_id = device_id

        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # initialize kernel
 
    def forward(self, embbedings, label):
        if self.device_id == None:
            kernel_norm = l2_norm(self.kernel, axis = 0)
            cos_theta = torch.mm(embbedings, kernel_norm)
        else:
            x = embbedings
            sub_kernels = torch.chunk(self.kernel, len(self.device_id), dim=1)
            temp_x = x.cuda(self.device_id[0])
            kernel_norm = l2_norm(sub_kernels[0], axis = 0).cuda(self.device_id[0])
            cos_theta = torch.mm(temp_x, kernel_norm)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                kernel_norm = l2_norm(sub_kernels[i], axis = 0).cuda(self.device_id[i])
                cos_theta = torch.cat((cos_theta, torch.mm(temp_x, kernel_norm).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return output


class AdaCos(nn.Module):
    r"""Implementation for "Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations"
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        s: scale of outputs
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

class MagFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    def __init__(self, feat_dim, num_class, margin_am=0.0, scale=32, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8, lamda=20):
        super(MagFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_am = margin_am
        self.scale = scale        
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.lamda = lamda

    def calc_margin(self, x):
        margin = (self.u_margin-self.l_margin) / \
                 (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def forward(self, feats, labels):
        x_norm = torch.norm(feats, dim=1, keepdim=True).clamp(self.l_a, self.u_a)# l2 norm
        ada_margin = self.calc_margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        loss_g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm) 
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        min_cos_theta = torch.cos(math.pi - ada_margin)        
        cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output, self.lamda*loss_g

class MV_Softmax(Module):
    """Implementation for "Mis-classified Vector Guided Softmax Loss for Face Recognition"
    """
    def __init__(self, feat_dim, num_class, is_am, margin=0.35, mv_weight=1.12, scale=32):
        super(MV_Softmax, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.mv_weight = mv_weight
        self.scale = scale
        self.is_am = is_am
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = self.sin_m * margin

    def forward(self, x, label):
        kernel_norm = F.normalize(self.weight, dim=0)
        x = F.normalize(x)
        cos_theta = torch.mm(x, kernel_norm)
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1) 
        if self.is_am:  # AM
            mask = cos_theta > gt - self.margin
            final_gt = torch.where(gt > self.margin, gt - self.margin, gt)
        else:  # arcface
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m 
            mask = cos_theta > cos_theta_m
            final_gt = torch.where(gt > 0.0, cos_theta_m, gt)
        # process hard example.
        hard_example = cos_theta[mask]
        cos_theta[mask] = self.mv_weight * hard_example + self.mv_weight - 1.0
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.scale
        return cos_theta

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