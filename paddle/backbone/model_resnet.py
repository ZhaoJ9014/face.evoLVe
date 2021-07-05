import paddle.nn as nn
import paddle
from paddle.nn import Linear, Conv2D, BatchNorm1D, BatchNorm2D, ReLU, Dropout, MaxPool2D, Sequential, Layer


# Support: ['ResNet_50', 'ResNet_101', 'ResNet_152']


def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""

    return Conv2D(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1)


def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""

    return Conv2D(in_planes, out_planes, kernel_size = 1, stride = stride)


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None,zero_init_residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2D(planes,weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        self.relu = ReLU()
        self.conv2 = conv3x3(planes, planes)
        if zero_init_residual:
            self.bn2 = BatchNorm2D(planes,weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0),regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        else:
            self.bn2 = BatchNorm2D(planes,weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None,zero_init_residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2D(planes,weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2D(planes,weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        if zero_init_residual:
            self.bn3 = BatchNorm2D(planes * self.expansion,weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0),regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        else:
            self.bn3 = BatchNorm2D(planes * self.expansion,weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(Layer):

    def __init__(self, input_size, block, layers, zero_init_residual = True):
        super(ResNet, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64
        self.zero_init_residual = zero_init_residual
        self.conv1 = Conv2D(3, 64, kernel_size = 7, stride = 2, padding = 3,weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingNormal()))
        self.bn1 = BatchNorm2D(64,weight_attr=paddle.ParamAttr(
            initializer=nn.initializer.Constant(value=1),
            regularizer=paddle.regularizer.L1Decay(0.0)
        ),bias_attr=paddle.ParamAttr(
            initializer=nn.initializer.Constant(value=0),
            regularizer = paddle.regularizer.L1Decay(0.0)
        ))
        self.relu = ReLU()
        self.maxpool = MaxPool2D(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.bn_o1 = BatchNorm2D(2048,weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0),regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))
        self.dropout = Dropout()
        if input_size[0] == 112:
            self.fc = Linear(2048 * 4 * 4, 512)
        else:
            self.fc = Linear(2048 * 8 * 8, 512)
        self.bn_o2 = BatchNorm1D(512,weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0),regularizer=paddle.regularizer.L1Decay(0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0)))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2D(planes * block.expansion,weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0)),bias_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L1Decay(0.0))),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,self.zero_init_residual))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,self.zero_init_residual))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_o1(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        x = self.bn_o2(x)

        return x


def ResNet_50(input_size, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ResNet_101(input_size, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def ResNet_152(input_size, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 8, 36, 3], **kwargs)

    return model
