import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms.functional as trans
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def par_flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        #### input Bparsing (real images)
        # dir_BP = '_Bparsing' if self.opt.label_nc == 0 else '_img'
        # self.dir_BP = os.path.join(opt.dataroot, opt.phase + dir_BP)
        # self.BP_paths = sorted(make_dataset(self.dir_BP))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            # if self.opt.isTrain:
            # A = trans.rotate(A,angle=params['degree'],fill=255)
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('L'))

            # BP_path = self.BP_paths[index]
            # BP = torch.from_numpy(np.load(BP_path))
            # BP = F.interpolate(BP.unsqueeze(0), size=([self.opt.loadSize, self.opt.loadSize]), mode='nearest').squeeze(0)  ### resize
            # if self.opt.isTrain:
            #     BP = BP[:, params['crop_pos'][1]:params['crop_pos'][1] + self.opt.fineSize,
            #          params['crop_pos'][0]:params['crop_pos'][0] + self.opt.fineSize]  ###crop
            #     if params['flip']:
            #         BP = par_flip(BP, 2)
            #
            # A_tensor = torch.cat([A_tensor, BP], 0)

        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            # B = trans.rotate(B, angle=params['degree'], fill=255)
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B.convert('L'))

        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'