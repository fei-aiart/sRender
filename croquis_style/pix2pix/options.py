#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-13
# @Author  : Jehovah
# @File    : options.py
# @Software: PyCharm


import os
import torch
import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument('--dataroot', default="/home/meimei/mayme/data/sketch_gray/",
                        help="path to images (should have subfolders trainA, trainB, valA, valB, etc)")
    parser.add_argument('--testroot', default="/home/meimei/mayme/data/s_17_noback/",
                        help="path to test images ")
    parser.add_argument('--gpuid', type=str, default='2', help='which gpu to use')
    parser.add_argument('--load_size', type=int, default=542, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=512, help='then crop to this size')
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--beta', type=int, default=0.5, help='momentum parameters bata1')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='with batchSize=1 equivalent to instance normalization.')
    parser.add_argument('--niter', type=int, default=600, help='number of epochs to train for')
    parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--lamb1', type=int, default=10, help='weight on VGG term in objective')
    parser.add_argument('--experiment', type=str, default='./experiment', help='models are saved here')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='image are saved here')
    parser.add_argument('--output', default='./output', help='folder to output images ')
    parser.add_argument('--result', default='./result', help='folder to test images ')
    parser.add_argument('--pretrain', default="/home/meimei/mayme/code/sketch_sys_model/net_G_595.pth",
                        help='folder to pretrianed model ')
    # parser.add_argument('--pretrain', default='/data/mayme/code/sketch_sys/checkpoints/net_G_500.pth',
    #                     help='folder to pretrianed model ')
    opt = parser.parse_args()
    return opt



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)




