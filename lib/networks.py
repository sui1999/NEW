import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import logging

from scipy import ndimage

from lib.pvtv2 import pvt_v2_b2, pvt_v2_b5, pvt_v2_b0
from lib.decoders import CUP, CASCADE, CASCADE_Cat, GCUP, GCUP_Cat, GCASCADE, GCASCADE_Cat, MyDecoderLayer
from lib.pyramid_vig import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu

from lib.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from lib.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from lib.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out

from torchvision.transforms.functional import rgb_to_grayscale
import math

logger = logging.getLogger(__name__)


class MyActFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        # X=[x1, x2, x3,x4] F=[P1, P2, P3, P4] dF=[F1, F2, F3, F4]
        # sL = 1.75的情况
        one = torch.ones_like(input_)
        zero = torch.zeros_like(input_)
        x1 = torch.where(input_ < -1, 0.03125 * input_ - 0.09375, zero)
        x2 = torch.where(input_ < -1, zero, input_)
        x2 = torch.where(x2 > 0, zero, x2)
        x3 = torch.where(input_ < 0, zero, input_)
        x3 = torch.where(x3 > 1, zero, x3)
        x4 = torch.where(input_ <= 1, -1 * one, input_)

        x2 = ((0.0625 * x2 + 0.25) * x2 + 0.3125) * x2

        x3 = ((-1.1875 * x3 + 1.875) * x3 + 0.3125) * x3
        x4 = 0.5 * x4 + 0.5
        return x1 + x2 + x3 + x4

    @staticmethod
    def backward(ctx, grad_output):
        # X=[x1, x2, x3,x4] F=[P1, P2, P3, P4] dF=[F1, F2, F3, F4]
        # sL = 1.75的情况
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        one = torch.ones_like(input_)
        zero = torch.zeros_like(input_)
        x1 = torch.where(input_ < -1, 0.03125 * one, zero)
        x2 = torch.where(input_ < -1, zero, input_)
        x2 = torch.where(x2 > 0, zero, x2)
        mask2 = x2 == 0
        mask2 = torch.where(mask2, 0, 1)
        x3 = torch.where(input_ < 0, -1 * one, input_)
        x3 = torch.where(x3 > 1, -1 * one, x3)
        mask3 = x3 == -1
        mask3 = torch.where(mask3, 0, 1)
        x4 = torch.where(input_ <= 1, zero, one)
        x2 = (0.1875 * x2 + 0.5) * x2 + 0.3125
        x3 = (-3.5625 * x3 + 3.75) * x3 + 0.3125
        x4 = 0.5
        return grad_input * (x1 + x2 * mask2 + x3 * mask3 + x4)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class PVT_CUP(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CUP, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # decoder initialization
        self.decoder = CUP(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
              ('CUP decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)

        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4


class PVT_CASCADE(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CASCADE, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # decoder initialization
        self.decoder = CASCADE(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
              ('CASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)

        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4


class PVT_CASCADE_Cat(nn.Module):
    def __init__(self, n_class=1):
        super(PVT_CASCADE_Cat, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        print('Model %s created, param count: %d' %
              ('PVT backbone: ', sum([m.numel() for m in self.backbone.parameters()])))

        # decoder initialization
        self.decoder = CASCADE_Cat(channels=[512, 320, 128, 64])

        print('Model %s created, param count: %d' %
              ('CASCADE_Cat decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)

        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4


class PVT_GCUP(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu',
                 skip_aggregation='additive'):
        super(PVT_GCUP, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.channels = [512, 320, 128, 64]

        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCUP(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv,
                                gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCUP_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv,
                                    gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1] * 2, self.channels[2] * 2, self.channels[3] * 2]
        else:
            print(
                'No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCUP(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv,
                                gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
              ('GCUP_decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)

        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4


class PVT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu',
                 skip_aggregation='additive'):
        super(PVT_GCASCADE, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.channels = [512, 320, 128, 64]

        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv,
                                    gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv,
                                        gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1] * 2, self.channels[2] * 2, self.channels[3] * 2]
        else:
            print(
                'No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv,
                                    gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
              ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)

        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4


class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,  # 栅格大小
            spline_order=3,  # 样条阶数
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=True):
        print("x111", x.shape)
        print("self.layers", self.layers)
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x


class Mp(nn.Module):
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
        # self.kan = KANLinear([in_features, hidden_features])
        act_relu = MyActFunc.apply
        if self.linear:
            # self.relu = nn.ReLU(inplace=True)
            self.relu = act_relu

    def forward(self, x):
        print("MLP的输入", x.shape)
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        print("MLP的输出", x.shape)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 替换卷积层为线性层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.kan1 = KANLinear(in_features, hidden_features)
        self.in_features = in_features  # 768
        self.hidden_features = hidden_features  # 3072
        self.dwconv = DWConvLKA(hidden_features)  # 保留这个深度卷积
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.kan2 = KANLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        act_relu = MyActFunc.apply
        if self.linear:
            self.relu = act_relu

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        x = self.kan1(x)  # 使用线性层
        x = x.permute(0, 3, 1, 2)
        if self.linear:
            x = self.relu(x)  # 如果是linear，则使用自定义激活函数
        x = self.dwconv(x)  # 深度卷积（假设它能处理展平后的数据）
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # Dropout
        x = x.permute(0, 2, 3, 1)
        x = self.kan2(x)  # 输出层
        x = x.permute(0, 3, 1, 2)
        x = self.drop(x)  # 再次使用 Dropout
        return x


def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


class SpatialAttentionBlock222(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock222, self).__init__()
        # act_relu = MyActFunc.apply
        # self.act_relu = act_relu

        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
            # act_relu
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
            # act_relu
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        # proj_query = self.act_relu(proj_query)
        proj_key = self.key(y).view(B, -1, W * H)
        # proj_key = self.act_relu(proj_key)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(y).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock222(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock222, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = y.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = y.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention222(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention222, self).__init__()
        self.sab = SpatialAttentionBlock222(in_channels)
        self.cab = ChannelAttentionBlock222(in_channels)

        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.mlp = Mlp(in_features=in_channels, hidden_features=4 * in_channels,
                       act_layer=nn.GELU, drop=0., linear=False)
        # self.skan = KANLinear(in_channels, 4 * in_channels)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        x1 = x + self.sab(x, x)
        x2 = x + self.cab(x, x)
        x11 = x1 + self.sab(x1, x2)
        x22 = x2 + self.cab(x2, x1)
        # out = torch.cat((x11, x22), dim=1)
        # out = self.conv1x1(out)
        out = x11 + x22 + x
        # print("out11", out.shape)
        out = self.mlp(out)
        # print("out22", out.shape)
        # out1 = self.skan(out)
        # print("out33", out1.shape)
        return out


# START Edge-Guided Attention Module
class ChannelGate_C(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate_C, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        # self.mlp = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(gate_channels, gate_channels // reduction_ratio),
        # )
        # self.Linear = nn.Linear(gate_channels // reduction_ratio, gate_channels)

    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate_C(nn.Module):
    def __init__(self):
        super(SpatialGate_C, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate_C(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate_C()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        # act_relu = MyActFunc.apply
        # self.act_relu = act_relu
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)
        self.conv_pred = nn.Conv2d(in_channels, 1, 1)

    def forward(self, edge_feature, x, pred):
        # print("edge_feature", edge_feature.shape)  # [1, 1, 112, 112]
        # print("x", x.shape)  # [1, 384, 14, 14]
        # print("pred", pred.shape)  # [1, 196, 384]
        B, N, C = pred.size()
        H = int(math.sqrt(N))
        pred = pred.permute(0, 2, 1)
        pred = pred.view(B, C, H, H)  # [1, 384, 14, 14]
        pred = self.conv_pred(pred)

        residual = x
        xsize = x.size()[2:]

        pred = torch.sigmoid(pred)

        # reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        # boudary attention
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        # high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        # fusion_feature = self.act_relu(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out


# END Edge-Guided Attention Module


class MERIT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256, 256), img_size_s2=(224, 224), k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu', interpolation='bicubic', skip_aggregation='additive'):
        super(MERIT_GCASCADE, self).__init__()

        self.interpolation = interpolation
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.skip_aggregation = skip_aggregation
        self.n_class = n_class

        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone1 = maxxvit_rmlp_small_rw_256_4out()  # [64, 128, 320, 512]
        # self.backbone2 = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]

        print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        state_dict1 = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        self.backbone1.load_state_dict(state_dict1, strict=False)
        #
        # print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        # state_dict2 = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        # self.backbone2.load_state_dict(state_dict2, strict=False)

        print('Pretrain weights loaded.')

        # # decoder initialization
        # if self.skip_aggregation == 'additive':
        #     self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv,
        #                              gcb_act=gcb_act, activation=activation)
        #     self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv,
        #                              gcb_act=gcb_act, activation=activation)
        # elif self.skip_aggregation == 'concatenation':
        #     self.decoder1 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding,
        #                                  conv=conv, gcb_act=gcb_act, activation=activation)
        #     # self.decoder2 = GCASCADE_Cat(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding,
        #     #                              conv=conv, gcb_act=gcb_act, activation=activation)
        #     self.channels = [self.channels[0], self.channels[1] * 2, self.channels[2] * 2, self.channels[3] * 2]
        # else:
        #     print(
        #         'No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
        #     self.decoder1 = GCASCADE(channels=self.channels, img_size=img_size_s1[0], k=k, padding=padding, conv=conv,
        #                              gcb_act=gcb_act, activation=activation)
        #     self.decoder2 = GCASCADE(channels=self.channels, img_size=img_size_s2[0], k=k, padding=padding, conv=conv,
        #                              gcb_act=gcb_act, activation=activation)
        #
        # print('Model %s created, param count: %d' %
        #       ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder1.parameters()])))
        # # print('Model %s created, param count: %d' %
        # #       ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder2.parameters()])))

        d_base_feat_size = 8  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384],
            [768, 768, 768, 768, 768],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        self.AffinityAttention222 = AffinityAttention222(768)
        self.ega3 = EGA(384)
        self.ega2 = EGA(192)
        self.ega1 = EGA(96)
        self.decoder_3 = MyDecoderLayer(
            (d_base_feat_size, d_base_feat_size),
            in_out_chan[3],
            n_class=n_class,
        )

        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            n_class=n_class,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            n_class=n_class,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            n_class=n_class,
            is_last=True,
        )

    def forward(self, x):
        # print("x", x.shape)
        # 12, 1, 256, 256
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)

        grayscale_img = rgb_to_grayscale(x)
        # print("灰度化", grayscale_img.shape)
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]
        # print("edge_feature", edge_feature.shape)
        # [12, 1, 128, 128]

        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        # print("f1[3]1", f1[3].shape)
        f1[3] = self.AffinityAttention222(f1[3])
        # print("f1[3]2", f1[3].shape)

        # decoder
        b, c, _, _ = f1[3].shape

        tmp_3 = self.decoder_3(f1[3].permute(0, 2, 3, 1).view(b, -1, c))
        # print("tmp_3", tmp_3.shape)
        ega3 = self.ega3(edge_feature, f1[2], tmp_3)
        # print("ega3", ega3.shape)
        tmp_2 = self.decoder_2(tmp_3, ega3.permute(0, 2, 3, 1))
        # print("tmp_2", tmp_2.shape)
        ega2 = self.ega2(edge_feature, f1[1], tmp_2)
        # print("ega2", ega2.shape)
        tmp_1 = self.decoder_1(tmp_2, ega2.permute(0, 2, 3, 1))
        # print("tmp_1", tmp_1.shape)
        ega1 = self.ega1(edge_feature, f1[0], tmp_1)
        # print("ega1", ega1.shape)
        tmp_0 = self.decoder_0(tmp_1, ega1.permute(0, 2, 3, 1))
        # print("tmp_0", tmp_0.shape)

        return tmp_0


if __name__ == '__main__':
    model = PVT_GCASCADE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())
