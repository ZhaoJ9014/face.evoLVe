from __future__ import print_function
from __future__ import division
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


# Support: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']

class Softmax(nn.Layer):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
        """

    def __init__(self, in_features, out_features):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_arr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant())
        self.linear = paddle.nn.Linear(in_features,out_features,weight_attr=weight_arr, bias_attr=bias_attr)

    def forward(self, x):
        out = self.linear(x)
        return out


class ArcFace(nn.Layer):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        weight_arr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        self.linear = paddle.nn.Linear(in_features, out_features, weight_attr=weight_arr)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        self.linear.weight.Tensor = F.normalize(self.linear.weight)
        x = F.normalize(input)
        cosine = self.linear(x)
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        label = label.astype(dtype='int64').flatten()
        one_hot = F.one_hot(label, num_classes=phi.shape[1])
        assert one_hot.shape == cosine.shape , 'shape error'
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class CosFace(nn.Layer):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        weight_arr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        self.linear = paddle.nn.Linear(in_features,out_features, weight_attr=weight_arr)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        self.linear.weight.Tensor = F.normalize(self.linear.weight)
        x = F.normalize(input)
        cosine = self.linear(x)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        label = label.astype(dtype='int64').flatten()
        one_hot = F.one_hot(label, num_classes=phi.shape[1])
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'


class SphereFace(nn.Layer):
    r"""Implement of SphereFace (https://arxiv.org/pdf/1704.08063.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, m=4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0

        weight_arr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        self.linear = paddle.nn.Linear(in_features,out_features, weight_attr=weight_arr)

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
        self.linear.weight.Tensor = F.normalize(self.linear.weight)
        x = F.normalize(input)
        cos_theta = self.linear(x)
        cos_theta = cos_theta.clip(min=-1,max=1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.acos()
        k = paddle.floor(self.m * theta / 3.14159265)
        phi_theta = paddle.to_tensor(((-1.0) ** k) * cos_m_theta - 2 * k)
        NormOfFeature = paddle.norm(input, p=2, axis=1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = F.one_hot(label, num_classes=phi_theta.shape[1])
        one_hot = paddle.reshape(one_hot,(phi_theta.shape[0], phi_theta.shape[1]))
        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.reshape((-1, 1))

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', m = ' + str(self.m) + ')'


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.divide(input, norm)

    return output


class Am_softmax(nn.Layer):
    r"""Implement of Am_softmax (https://arxiv.org/pdf/1801.05599.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        s: scale of outputs
    """

    def __init__(self, in_features, out_features, m=0.35, s=30.0):
        super(Am_softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        weight_arr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        self.linear = paddle.nn.Linear(in_features,out_features, weight_attr=weight_arr)
        self.linear.weight.Tensor = paddle.norm(self.linear.weight, p=2, axis=1).clip(max=1e-5)*1e5
        # ###self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # initialize kernel

    def forward(self, embbedings, label):
        self.linear.weight.Tensor = l2_norm(self.linear.weight, axis=0)
        cos_theta = self.linear(embbedings)
        cos_theta = paddle.clip(cos_theta,min=-1,max=1) # for numerical stability
        phi = cos_theta - self.m
        label = label.reshape((-1, 1))  # size=(B,1)
        index = paddle.to_tensor(cos_theta.numpy()[0] * 0.0 ) # size=(B,Classnum)
        index.scatter_(1, label.reshape((-1, 1)), 1)
        # ###index = index.byte()
        index = index.astpye(dtype='uint8')
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return output
