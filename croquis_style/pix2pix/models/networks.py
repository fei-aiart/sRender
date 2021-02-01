#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : networks.py
# @Author: Jehovah
# @Date  : 18-9-19
# @Desc  : 

import torch
import torch.nn as nn

# opt.norm=batch, not opt.no_dropout=True

class ResetBlock(nn.Module):
    def __init__(self, dim, stride=1):
        super(ResetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3,
                      stride=stride, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3,
                      stride=stride, bias=False),
            nn.InstanceNorm2d(dim),
        )


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

# class ResetBlock(nn.Module):
#     def __init__(self):
#         super(ResetBlock, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1),
#             nn.InstanceNorm1d(256),
#             nn.ReLU(True),
#             # nn.Dropout(0.5),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(256, 256, kernel_size=3),
#             nn.InstanceNorm2d(256),
#         )
#
#     def forward(self, x):
#         out = self.block1(x)
#         return out + x


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        self.cov1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=(1,1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1,1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.decov1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.decov2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.cov = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, stride=1),
            nn.Tanh()
        )
        se = []
        for i in range(9):
            se += [ResetBlock(dim=256)]
        self.res = nn.Sequential(*se)

    def forward(self, x):
        out = self.cov1(x)
        out = self.cov2(out)
        out = self.cov3(out)
        out = self.res(out)
        out = self.decov1(out)
        out = self.decov2(out)
        out = self.cov(out)
        return out



class Discriminator(nn.Module):
    def __init__(self,input_nc, output_nc, ndf=64):
        super(Discriminator,self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.cov4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.cov5 = nn.Sequential(
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),

        )

    def forward(self, x):
        out_cov1 = self.cov1(x)
        out_cov2 = self.cov2(out_cov1)
        out_cov3 = self.cov3(out_cov2)
        out_cov4 = self.cov4(out_cov3)
        out = self.cov5(out_cov4)
        return out

class DiscriminatorWGANGP(nn.Module):

    def __init__(self, in_dim,out_dim, dim=64):
        super(DiscriminatorWGANGP, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y