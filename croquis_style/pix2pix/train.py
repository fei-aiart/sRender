#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-13
# @Author  : Jehovah
# @File    : train.py
# @Software: PyCharm


import time
import options
from data_loader2 import DataLoader

from models.networks import *
from utils.utils import GANLoss
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision.utils as vutils
import os
from utils.vggloss import *
import numpy as np
import matplotlib.pyplot as plt

opt = options.init()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid  # 指定gpu
train_set = DataLoader(opt)
train_loader = data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=8)

test_set = DataLoader(opt, isTrain=False)
test_loader = data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=False, num_workers=8)


net_G = Generator(opt.input_nc, opt.output_nc).cuda()
net_D = Discriminator(opt.input_nc, opt.output_nc).cuda()

# vgg = vgg_load().cuda()
# criterionVGG = Perceptualloss(vgg)
criterionGAN = GANLoss()
critertionL1 = nn.L1Loss()

# initialize optimizers
optimizerG = torch.optim.Adam(net_G.parameters(),lr=opt.lr, betas=(opt.beta, 0.999))
optimizerD = torch.optim.Adam(net_D.parameters(),lr=opt.lr, betas=(opt.beta, 0.999))
optimizers = []
schedulers = []



def train():

    for epoch in range(opt.niter):

        epoch_start_time = time.time()
        for i, (ima,imb) in enumerate(train_loader):
            img_A = ima.cuda()
            img_B = imb.cuda()
            real_A = img_A
            real_B = img_B

            fake_B = net_G(real_A)
            net_D.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = net_D(fake_AB.detach())

            loss_D_fake = criterionGAN(pred_fake, False)

            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = net_D(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()

            optimizerD.step()

            # netG
            net_G.zero_grad()

            fake_AB = torch.cat((real_A, fake_B), 1)
            out_put = net_D(fake_AB)
            loss_G_GAN = criterionGAN(out_put, True)
            loss_G_L1 = critertionL1(fake_B, real_B) * opt.lamb

            # b, c, w, h = fake_B.shape
            # fake = fake_B.expand(b, 3, w, h)
            # real = real_A.expand(b, 3, w, h)
            #
            # loss_G_VGG = criterionVGG(fake, real) * opt.lamb1
            # loss_G = loss_G_GAN + loss_G_L1 + loss_G_VGG
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizerG.step()
            if i % 100 == 0:
                print('[%d/%d][%d/%d] LOSS_D: %.4f LOSS_G: %.4f LOSS_L1: %.4f' % (
                epoch, opt.niter, i, len(train_loader), loss_D, loss_G, loss_G_L1))
                print('LOSS_real: %.4f LOSS_fake: %.4f' % (loss_D_real, loss_D_fake))

        print('Time Taken: %d sec' % (time.time() - epoch_start_time))
        # if epoch % 5 == 0:
        #
        #     vutils.save_image(real_A.data,
        #                       './sample/fake_samples_epoch_%03d.png' % (epoch),
        #                       normalize=True)
        #     vutils.save_image(fake_B.data,
        #                       './sample/fake_samples_epoch_%03d.png' % (epoch),
        #                       normalize=True)
        if epoch % 5 == 0:
            test(epoch, net_G, test_loader)
        if epoch > 100 and epoch % 5 ==0:
            print("save net")

            if not os.path.exists(opt.checkpoints):
                os.makedirs(opt.checkpoints)
            torch.save(net_G.state_dict(), opt.checkpoints + '/net_G_{}.pth'.format(epoch))
            torch.save(net_D.state_dict(), opt.checkpoints + '/net_D_{}.pth'.format(epoch))


def test(epoch, netG_A, test_data):

    save_dir_B = opt.output + '/B_' + str(epoch)


    mkdir(save_dir_B)

    for i,(ima,imb) in enumerate(test_data):
        if i <10:
            img_A = ima
            img_B = imb


            real_A = img_A.cuda()
            real_B = img_B.cuda()
            fake_B = netG_A(real_A)
            img = torch.cat([real_A,fake_B,real_B],0)


            output_name_B = '{:s}/{:s}{:s}'.format(
                save_dir_B, str(i + 1), '.jpg')


            vutils.save_image(img, output_name_B, normalize=True, scale_each=True)

    print(str(epoch) + " saved")

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    train()
