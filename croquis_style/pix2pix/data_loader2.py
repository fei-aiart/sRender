import random
import torch
import torch.utils.data as data
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import PIL
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def default_loader(path):
    return Image.open(path).convert('RGB')

def flip(img, if_flip):
    if if_flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def par_flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class DataLoader(data.Dataset):
    def __init__(self, opt, isTrain=True, transform=None, return_paths=None, loader=default_loader):
        super(DataLoader, self).__init__()
        self.opt = opt
        self.isTrain = isTrain
        if isTrain:
            self.Alist = os.listdir(os.path.join(self.opt.dataroot,'train_A'))
            self.Blist = os.listdir(os.path.join(self.opt.dataroot,'train_B'))

        else:
            print( "test start :")
            self.Alist = os.listdir(os.path.join(self.opt.dataroot, 'test_A'))
            self.Blist = os.listdir(os.path.join(self.opt.dataroot, 'test_B'))

        self.A_size = len(self.Alist)  # get the size of dataset A
        self.B_size = len(self.Blist)  # get the size of dataset B
        print("A size:", self.A_size)
        print("B size:", self.B_size)
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):

        if self.isTrain:
            Apath = os.path.join(self.opt.dataroot,'train_A',self.Alist[index])
            Bpath = os.path.join(self.opt.dataroot,'train_B',Apath.split('/')[-1])
        else:
            Apath = os.path.join(self.opt.dataroot, 'test_A', self.Alist[index])
            Bpath = os.path.join(self.opt.dataroot, 'test_B', Apath.split('/')[-1])

        imgA = Image.open(Apath).convert('L')
        imgB = Image.open(Bpath).convert('L')

        if self.isTrain:

            imgA = transforms.Resize((self.opt.crop_size, self.opt.crop_size), interpolation=PIL.Image.NEAREST)(imgA)
            imgB = transforms.Resize((self.opt.crop_size, self.opt.crop_size), interpolation=PIL.Image.NEAREST)(imgB)
            w, h = imgA.size
            pading_w = (self.opt.load_size - w) // 2
            pading_h = (self.opt.load_size - h) // 2
            padding = transforms.Pad((pading_w, pading_h), fill=0, padding_mode='constant')

            i = random.randint(0, self.opt.load_size - self.opt.crop_size)
            j = random.randint(0, self.opt.load_size - self.opt.crop_size)
            if_flip_h = random.random() > 0.5
            r = random.randint(0,90)

            imgA = self.process_img(imgA, i, j, padding, if_flip_h, r)
            imgB = self.process_img(imgB, i, j, padding, if_flip_h, r)


        else:

            imgA = imgA.convert('L')
            imgA = transforms.Resize((self.opt.crop_size, self.opt.crop_size), interpolation=PIL.Image.NEAREST)(imgA)
            imgA = transforms.ToTensor()(imgA)
            imgA = transforms.Normalize([0.5], [0.5])(imgA)

            imgB = imgB.convert('L')
            imgB = transforms.Resize((self.opt.crop_size, self.opt.crop_size), interpolation=PIL.Image.NEAREST)(imgB)
            imgB = transforms.ToTensor()(imgB)
            imgB = transforms.Normalize([0.5], [0.5])(imgB)

        return imgA, imgB

    def __len__(self):
        return len(self.Alist)

    def process_img(self, img, i, j,padding, if_flip_h,r):
        img = padding(img)
        img = img.crop((j, i, j + self.opt.crop_size, i + self.opt.crop_size))
        if if_flip_h:
            img = tf.hflip(img)
        # img = tf.rotate(img,angle=r,fill=255).convert('L')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.5], [0.5])(img)
        return img