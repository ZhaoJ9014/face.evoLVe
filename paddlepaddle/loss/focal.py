import paddle
import paddle.nn as nn


# Support: ['FocalLoss']


class FocalLoss(nn.Layer):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target.astype('int64'))
        p = paddle.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
