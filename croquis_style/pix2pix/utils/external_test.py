
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL
from PIL import Image
import os
import os.path

def default_loader(path):
    return Image.open(path).convert('RGB')

class DataLoader(data.Dataset):
    def __init__(self, opt, isTrain=True, transform=None, return_paths=None, loader=default_loader):
        super(DataLoader, self).__init__()
        self.opt = opt
        self.isTrain = isTrain

        self.Alist = os.listdir(os.path.join(self.opt.testroot, 'test_A'))
        self.Blist = os.listdir(os.path.join(self.opt.testroot, 'test_B'))
        self.A_size = len(self.Alist)  # get the size of dataset A
        self.B_size = len(self.Blist)  # get the size of dataset A

        print("A size:", self.A_size)


        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        Apath = os.path.join(self.opt.testroot, 'test_A', self.Alist[index])
        Bpath = os.path.join(self.opt.testroot, 'test_B', Apath.split('/')[-1])

        imgA = Image.open(Apath).convert('L')
        imgB = Image.open(Bpath).convert('RGB')

        imgA = transforms.Resize((self.opt.crop_size, self.opt.crop_size), interpolation=PIL.Image.NEAREST)(imgA)
        imgA = transforms.ToTensor()(imgA)
        imgA = transforms.Normalize([0.5], [0.5])(imgA)

        imgB = transforms.Resize((self.opt.crop_size, self.opt.crop_size), interpolation=PIL.Image.NEAREST)(imgB)
        imgB = transforms.ToTensor()(imgB)
        imgB = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(imgB)


        return imgA, imgB,Apath.split('/')[-1]

    def __len__(self):
        return len(self.Alist)

