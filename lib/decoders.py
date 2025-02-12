import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

import math
from PIL import Image
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc

from lib.gcn_lib import Grapher as GCB

from einops import rearrange
from einops.layers.torch import Rearrange


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UCB(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1, activation='relu'):
        super(UCB, self).__init__()

        if (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif (activation == 'gelu'):
            self.activation = nn.GELU()
        elif (activation == 'relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif (activation == 'hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(ch_in),
            self.activation,
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class trans_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=4, stride=2, padding=1, groups=32):
        super(trans_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                               bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        activation = 'relu'
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            # nn.GroupNorm(1,1),
            nn.Sigmoid()
        )

        if (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif (activation == 'gelu'):
            self.activation = nn.GELU()
        elif (activation == 'relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif (activation == 'hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        activation = 'relu'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)

        if (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif (activation == 'gelu'):
            self.activation = nn.GELU()
        elif (activation == 'relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif (activation == 'hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        # print(x.shape)
        max_pool_out = self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)


class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CUP(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64]):
        super(CUP, self).__init__()

        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.ConvBlock3 = conv_block(ch_in=2 * channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.ConvBlock2 = conv_block(ch_in=2 * channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.ConvBlock1 = conv_block(ch_in=2 * channels[3], ch_out=channels[3])

    def forward(self, x, skips):
        d4 = self.ConvBlock4(x)

        # decoding + concat path
        d3 = self.Up3(d4)
        d3 = torch.cat((skips[0], d3), dim=1)

        d3 = self.ConvBlock3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((skips[1], d2), dim=1)
        d2 = self.ConvBlock2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((skips[2], d1), dim=1)
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64]):
        super(CASCADE_Cat, self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2 * channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2 * channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2))
        self.ConvBlock1 = conv_block(ch_in=2 * channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2 * channels[1])
        self.CA2 = ChannelAttention(2 * channels[2])
        self.CA1 = ChannelAttention(2 * channels[3])

        self.SA = SPA()

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        # upconv3
        d3 = self.Up3(d4)

        # AG3
        x3 = self.AG3(g=d3, x=skips[0])

        # Concat 3
        d3 = torch.cat((x3, d3), dim=1)

        # CAM3
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        # upconv2
        d2 = self.Up2(d3)

        # AG2
        x2 = self.AG2(g=d2, x=skips[1])

        # Concat 2
        d2 = torch.cat((x2, d2), dim=1)

        # CAM2
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        # print(d2.shape)
        d2 = self.ConvBlock2(d2)

        # upconv1
        d1 = self.Up1(d2)

        # print(skips[2])
        # AG1
        x1 = self.AG1(g=d1, x=skips[2])

        # Concat 1
        d1 = torch.cat((x1, d1), dim=1)

        # CAM1
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class CASCADE(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64]):
        super(CASCADE, self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])

        self.SA = SPA()

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        # upconv3
        d3 = self.Up3(d4)

        # AG3
        x3 = self.AG3(g=d3, x=skips[0])

        # Concat 3
        d3 = d3 + x3

        # CAM3
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        # upconv2
        d2 = self.Up2(d3)

        # AG2
        x2 = self.AG2(g=d2, x=skips[1])

        # Concat 2
        d2 = d2 + x2

        # CAM2
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        # print(d2.shape)
        d2 = self.ConvBlock2(d2)

        # upconv1
        d1 = self.Up1(d2)

        # print(skips[2])
        # AG1
        x1 = self.AG1(g=d1, x=skips[2])

        # Concat 1
        d1 = d1 + x1

        # CAM1
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1


class GCUP(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], img_size=224, drop_path_rate=0.0, k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu'):
        super(GCUP, self).__init__()

        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation

        #  Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act  # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch'  # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path, self.drop_path, self.drop_path, self.drop_path]  # stochastic depth decay rule
        self.num_knn = [self.k, self.k, self.k, self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4

        self.gcb4 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.act, self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4),
                drop_path=self.dpr[0],
                relative_pos=True, padding=self.padding),
        )

        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(
            GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4),
                drop_path=self.dpr[1],
                relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1], ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(
            GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // (4),
                drop_path=self.dpr[2],
                relative_pos=True, padding=self.padding),
        )

        self.ucb1 = UCB(ch_in=channels[2], ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(
            GCB(channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                relative_pos=True, padding=self.padding),
        )

    def forward(self, x, skips):
        # GCAM4
        d4 = self.gcb4(x)

        # UCB3
        d3 = self.ucb3(d4)

        # Aggregation 3
        d3 = d3 + skips[0]

        # GCAM3
        d3 = self.gcb3(d3)

        # UCB2
        d2 = self.ucb2(d3)

        # Aggregation 2
        d2 = d2 + skips[1]

        # GCAM2
        d2 = self.gcb2(d2)

        # UCB1
        d1 = self.ucb1(d2)

        # Aggregation 1
        d1 = d1 + skips[2]

        # GCAM1
        d1 = self.gcb1(d1)

        return d4, d3, d2, d1


class GCUP_Cat(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], img_size=224, drop_path_rate=0.0, k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu'):
        super(GCUP_Cat, self).__init__()

        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation

        #  Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act  # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch'  # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path, self.drop_path, self.drop_path, self.drop_path]  # stochastic depth decay rule
        self.num_knn = [self.k, self.k, self.k, self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4

        self.gcb4 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4),
                drop_path=self.dpr[0],
                relative_pos=True, padding=self.padding),
        )

        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(
            GCB(2 * channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4),
                drop_path=self.dpr[1],
                relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=2 * channels[1], ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(
            GCB(2 * channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // (4),
                drop_path=self.dpr[2],
                relative_pos=True, padding=self.padding),
        )

        self.ucb1 = UCB(ch_in=2 * channels[2], ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(
            GCB(2 * channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                relative_pos=True, padding=self.padding),
        )

    def forward(self, x, skips):
        # GCAM4
        d4 = self.gcb4(x)

        # UCB3
        d3 = self.ucb3(d4)

        # Aggregation 3
        d3 = torch.cat((skips[0], d3), dim=1)

        # GCAM3
        d3 = self.gcb3(d3)

        # UCB2
        d2 = self.ucb2(d3)

        # Aggregation 2
        d2 = torch.cat((skips[1], d2), dim=1)

        # GCAM2
        d2 = self.gcb2(d2)

        # UCB1
        d1 = self.ucb1(d2)

        # Aggregation 1
        d1 = torch.cat((skips[2], d1), dim=1)

        # GCAM1
        d1 = self.gcb1(d1)

        return d4, d3, d2, d1


class GCASCADE(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], drop_path_rate=0.0, img_size=224, k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu'):
        super(GCASCADE, self).__init__()

        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation

        #  Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act  # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch'  # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path, self.drop_path, self.drop_path, self.drop_path]  # stochastic depth decay rule
        self.num_knn = [self.k, self.k, self.k, self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4

        self.gcb4 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4),
                drop_path=self.dpr[0],
                relative_pos=True, padding=self.padding),
        )

        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(
            GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4),
                drop_path=self.dpr[1],
                relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1], ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(
            GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // (4),
                drop_path=self.dpr[2],
                relative_pos=True, padding=self.padding),
        )

        self.ucb1 = UCB(ch_in=channels[2], ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(
            GCB(channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                relative_pos=True, padding=self.padding),
        )

        self.spa = SPA()

    def forward(self, x, skips):
        # GCAM4
        d4 = self.gcb4(x)
        d4 = self.spa(d4) * d4

        # UCB3
        d3 = self.ucb3(d4)

        # Aggregation 3
        d3 = d3 + skips[0]  # torch.cat((skips[0],d3),dim=1)

        # GCAM3
        d3 = self.gcb3(d3)
        d3 = self.spa(d3) * d3

        # UCB2
        d2 = self.ucb2(d3)

        # Aggregation 2
        d2 = d2 + skips[1]  # torch.cat((skips[1],d2),dim=1)

        # GCAM2
        d2 = self.gcb2(d2)
        d2 = self.spa(d2) * d2

        # UCB1
        d1 = self.ucb1(d2)

        # Aggregation 1
        d1 = d1 + skips[2]  # torch.cat((skips[2],d1),dim=1)

        # GCAM1
        d1 = self.gcb1(d1)
        d1 = self.spa(d1) * d1

        return d4, d3, d2, d1


class GCASCADE_Cat(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], drop_path_rate=0.0, img_size=224, k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu'):
        super(GCASCADE_Cat, self).__init__()

        #  Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation

        #  Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act  # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch'  # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path, self.drop_path, self.drop_path, self.drop_path]  # stochastic depth decay rule
        self.num_knn = [self.k, self.k, self.k, self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4

        self.gcb4 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4),
                drop_path=self.dpr[0],
                relative_pos=True, padding=self.padding),
        )

        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(
            GCB(channels[1] * 2, self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4),
                drop_path=self.dpr[1],
                relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1] * 2, ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(
            GCB(channels[2] * 2, self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // (4),
                drop_path=self.dpr[2],
                relative_pos=True, padding=self.padding),
        )

        self.ucb1 = UCB(ch_in=channels[2] * 2, ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride,
                        padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(
            GCB(channels[3] * 2, self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                relative_pos=True, padding=self.padding),
        )

        self.spa = SPA()

    def forward(self, x, skips):
        # GCAM4
        d4 = self.gcb4(x)
        d4 = self.spa(d4) * d4

        # UCB3
        d3 = self.ucb3(d4)

        # Aggregation 3
        d3 = torch.cat((skips[0], d3), dim=1)

        # GCAM3
        d3 = self.gcb3(d3)
        d3 = self.spa(d3) * d3

        # ucb2
        d2 = self.ucb2(d3)

        # Aggregation 2
        d2 = torch.cat((skips[1], d2), dim=1)

        # GCAM2
        d2 = self.gcb2(d2)
        d2 = self.spa(d2) * d2

        # ucb1
        d1 = self.ucb1(d2)

        # Aggregation 1
        d1 = torch.cat((skips[2], d1), dim=1)

        # GCAM1
        d1 = self.gcb1(d1)
        d1 = self.spa(d1) * d1
        # print("d4", d4.shape)
        # print("d3", d3.shape)
        # print("d2", d2.shape)
        # print("d1", d1.shape)
        # torch.Size([12, 768, 8, 8])
        # torch.Size([12, 768, 16, 16])
        # torch.Size([12, 384, 32, 32])
        # torch.Size([12, 192, 64, 64])

        return d4, d3, d2, d1


# Decoder

class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)  # build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        #                       * self.attn(self.norm1(x)))

        y = x.permute(0, 2, 3, 1)  # b h w c, because norm requires this
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)  # b c h w, because attn requieres this
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
        #                       * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        # print("LKA return shape: {}".format(x.shape))
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
            self, input_size, in_out_chan, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        # print("Dim: {} | Out_dim: {} | Key_dim: {} | Value_dim: {} | X1_dim: {}".format(dims, out_dim, key_dim,
        # value_dim, x1_dim))
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # self.lka_attn = LKABlock(dim=dims) # TODO: Further input parameters here
            # self.cross_attn = CrossAttentionBlock( # Skip connection block
            #    dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            # )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        # self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_lka_1 = LKABlock(dim=out_dim)  # TODO
        # self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_lka_2 = LKABlock(dim=out_dim)  # TODO

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape  # 1 28 28 320, 1 56 56 128
            x2 = x2.view(b, -1, c)  # 1 784 320, 1 3136 128
            x1_expand = self.x1_linear(x1)  # 1 784 256 --> 1 784 320, 1 3136 160 --> 1 3136 128

            # cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2)) # 1 784 320, 1 3136 128
            cat_linear_x = x1_expand + x2  # Simply add them in the first step. TODO: Add more complex skip connection here

            # tran_layer_1 = self.layer_former_1(cat_linear_x, h, w) # 1 784 320, 1 3136 128
            tran_layer_1 = self.layer_lka_1(cat_linear_x, h, w)
            # tran_layer_2 = self.layer_former_2(tran_layer_1, h, w) # 1 784 320, 1 3136 128
            tran_layer_2 = self.layer_lka_2(tran_layer_1, h, w)  # LEON CHANGE THIS!

            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))  # 1 9 224 224
            else:
                out = self.layer_up(tran_layer_2)  # 1 3136 160
        else:
            out = self.layer_up(x1)
        return out
