#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-13
# @Author  : Jehovah
# @File    : test.py
# @Software: PyCharm

import options
from utils.external_test import DataLoader
from models.networks import *
import torch.utils.data as data
import torchvision.utils as vutils
import os
import torchvision.transforms as transform
import PIL

opt = options.init()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid  # 指定gpu

test_set = DataLoader(opt, isTrain=False)
test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

netG_A = Generator(opt.input_nc, opt.output_nc).cuda()
netG_A.load_state_dict(torch.load(opt.pretrain))

def test():
    save_dir = opt.result
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fine_dir = os.path.join(opt.output,'last')
    if not os.path.exists(fine_dir):
        os.makedirs(fine_dir)

    for i, (ima, imb,name) in enumerate(test_loader):

        print(name)
        img_A = ima
        img_B = imb

        real_A = img_A.cuda()
        real_B = img_B.cuda()
        fake_B = netG_A(real_A)

        # b, c, w, h = real_B.shape
        # real_A = real_A.expand(b, 3, w, h)
        # fake_B = fake_B.expand(b, 3, w, h)
        # img = torch.cat([real_A, fake_B, real_B], 0)

        output_name_B = '{:s}/{:s}'.format(save_dir, name[0])

        vutils.save_image(fake_B, output_name_B, normalize=True, scale_each=True)



if __name__ == '__main__':
    test()
