import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# @Author : Labyrinthine Leo
# @Time   : 2020.12.29

INPLACE=False

# -------------------- operations --------------------
class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):

        super(ReLUConvBN, self).__init__()

        self.relu = nn.ReLU(inplace=INPLACE)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, bn_train=False):
        x = self.relu(x)
        x = self.conv(x)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(Conv, self).__init__()

        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride  #
        self.padding = padding
        self.affine = affine

        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        x = self.ops(x)
        return x


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):

        super(SepConv, self).__init__()
        self.op = nn.Sequential(

            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),

            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, shape, affine=True):
        super(DilSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class AvgPool(nn.Module):
    def __init__(self, C_in, C_out, shape, kernel_size, stride=1, padding=0, count_include_pad=False):
        super(AvgPool, self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(padding, tuple):
            padding = 0

        self.op = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)

    def forward(self, x):
        if isinstance(self.padding, tuple):
            x = F.pad(x, self.padding)
        x = self.op(x)
        return x


class MaxPool(nn.Module):
    def __init__(self, C_in, C_out, shape, kernel_size, stride=1, padding=0):
        super(MaxPool, self).__init__()
        self.shape = shape
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(padding, tuple):
            padding = 0

        self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if isinstance(self.padding, tuple):
            x = F.pad(x, self.padding)
        x = self.op(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//
# Proceedings of the IEEE conference on computer vision and pattern recognition.
# 2018: 7132-7141.
# SENet
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# fun     : Bilinear Pooling Attention Module in pytorch
# @Author : Labyrinthine Leo
# @Time   : 2021.08.01
class BPAttLayer(nn.Module):
    def __init__(self, channel, reduction=16):

        super(BPAttLayer, self).__init__()
        self.re = reduction
        self.re_channel = channel // reduction
        self.sc = nn.Sequential(
            nn.Conv2d(channel, self.re_channel, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ec = nn.Sequential(
            nn.Linear(self.re_channel ** 2, (self.re_channel + 2) * self.re_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((self.re_channel + 2) * self.re_channel, channel, bias=False),
            nn.Sigmoid()

        )

    def forward(self, x):

        b, c, _, _ = x.shape
        x_ = self.sc(x).view(b, self.re_channel, -1)
        y = torch.bmm(x_, torch.transpose(x_, 1, 2))
        y = y.view(b, self.re_channel ** 2)
        y = self.ec(y).view(b, c, 1, 1)

        return x * y


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


# drop path operation
def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        #x.div_(drop_path_keep_prob)
        #x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()

        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
        
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        super(AuxHeadImageNet, self).__init__()

        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
    
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, shape, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0 # C_out为偶数
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False)) #
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path1 = x # [16, 32, 32, 60]
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        hw = [layer[0] for layer in layers] # [32, 16]
        c = [layer[-1] for layer in layers] # [60, 120]

        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]

        # previous reduction cell
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1] # h[i-1] = h[i] * 2
            self.relu = nn.ReLU(inplace=INPLACE)
            self.preprocess_x = FactorizedReduce(c[0], channels, [hw[0], hw[0], c[0]], affine)
            x_out_shape = [hw[1], hw[1], channels]
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0, [hw[0], hw[0]], affine)
            x_out_shape = [hw[0], hw[0], channels]
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(c[1], channels, 1, 1, 0, [hw[1], hw[1]], affine)
            y_out_shape = [hw[1], hw[1], channels]
            
        self.out_shape = [x_out_shape, y_out_shape]
    
    def forward(self, s0, s1, bn_train=False):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1, bn_train=bn_train)
        return [s0, s1]


class FinalCombine(nn.Module):
    def __init__(self, layers, out_hw, channels, concat, affine=True):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat = concat
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        for i in concat:
            hw = layers[i][0]
            if hw > out_hw:
                assert hw == 2 * out_hw and i in [0,1]
                self.concat_fac_op_dict[i] = len(self.ops)
                op = FactorizedReduce(layers[i][-1], channels, [hw, hw], affine)
                self.ops.append(op)
        
    def forward(self, states, bn_train=False):
        for i in self.concat:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i], bn_train)
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out

Operations_11 = {
    0:lambda c_in, c_out, stride, shape, affine: Identity(),

    1: lambda c_in, c_out, stride, shape, affine: Conv(c_in, c_out, 3, stride, 1, shape, affine=affine),
    2: lambda c_in, c_out, stride, shape, affine: Conv(c_in, c_out, 5, stride, 2, shape, affine=affine),

    3: lambda c_in, c_out, stride, shape, affine: SepConv(c_in, c_out, 3, stride, 1, shape, affine=affine),
    4: lambda c_in, c_out, stride, shape, affine: SepConv(c_in, c_out, 5, stride, 2, shape, affine=affine),

    5: lambda c_in, c_out, stride, shape, affine: Conv(c_in, c_out, (1,7), stride, ((0,3),(3,0)), shape, affine=affine),

    6: lambda c_in, c_out, stride, shape, affine: MaxPool(c_in, c_out, shape, kernel_size=3, stride=stride, padding=1),
    7: lambda c_in, c_out, stride, shape, affine: MaxPool(c_in, c_out, shape, kernel_size=5, stride=stride, padding=2),

    8: lambda c_in, c_out, stride, shape, affine: AvgPool(c_in, c_out, shape, kernel_size=3, stride=stride, padding=1),
    9: lambda c_in, c_out, stride, shape, affine: AvgPool(c_in, c_out, shape, kernel_size=5, stride=stride, padding=2),

    10:lambda c_in, c_out, stride, shape, affine: BPAttLayer(c_in)
    }

Operations_name = [
    'Identity', # identity
    'Conv 3*3',
    'Conv 5*5',

    'DSConv 3*3',
    'DSConv 5*5',
    
    'Conv 1*7+7*1',

    'MaxPool 3*3',
    'MaxPool 5*5',

    'AvgPool 3*3',
    'AvgPool 5*5',

    'BPAttLayer', # Bilinear Pooling Attention Layer (Homologous)
]