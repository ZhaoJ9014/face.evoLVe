import matplotlib.pyplot as plt
from paddle.optimizer.lr import LinearWarmup
from datetime import datetime

plt.switch_backend('agg')


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.sublayers()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        layer_info = str(layer.__class__)
        if 'backbone' in layer_info:
            continue
        if 'container' in layer_info:
            continue
        else:
            if 'BatchNorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape((1, -1)).expand_as(pred))

    res = []
    for k in topk:
        correct_k = (correct.numpy())[:k].reshape((-1)).astype('float32').sum(0)
        res.append(correct_k * (100.0 / batch_size))

    return res


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer._parameter_list:
        params.optimize_attr['learning_rate'] = batch * init_lr / num_batch_warm_up


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def schedule_lr(optimizer, ):
    # after warm_up_lr LinearWarmup replaced by learning_rate
    if type(optimizer._learning_rate) is LinearWarmup:
        optimizer._learning_rate = optimizer._learning_rate.learning_rate / 10.
    else:
        optimizer._learning_rate = optimizer._learning_rate / 10.


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

    return paras_only_bn, paras_wo_bn
