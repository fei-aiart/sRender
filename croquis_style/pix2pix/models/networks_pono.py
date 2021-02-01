import torch
import torch.nn as nn

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

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        self.ms = True
        self.pono = True
        self.stat_convs = nn.ModuleList()

        self.cov1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.cov2 = nn.Sequential(
            Conv2dBlock(64, 128, 'conv', use_bias=False, pono=self.pono),
        )
        self.cov3 = nn.Sequential(
            Conv2dBlock(128, 256, 'conv', use_bias=False, pono=self.pono),
        )

        up_model = []
        up_model += [Conv2dBlock(256, 128, 'deconv', use_bias=False, ms=self.ms)]
        up_model += [Conv2dBlock(128, 64, 'deconv', use_bias=False, ms=self.ms)]


        self.net1 = nn.Sequential(
            Conv2dBlock(2, 128, 'deconv', use_bias=False),
            Conv2dBlock(128, 256, 'stat_convs', use_bias=False, norm=False)
        )
        self.stat_convs.append(self.net1)
        self.net2 = nn.Sequential(
            Conv2dBlock(2, 64, 'deconv', use_bias=False),
            Conv2dBlock(64, 128, 'stat_convs', use_bias=False, norm=False)
        )
        self.stat_convs.append(self.net2)

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
        x = self.cov1(x)

        stats = []
        x, mean, std = self.cov2(x)
        stats.append((mean, std))

        x, mean, std = self.cov3(x)
        stats.append((mean, std))
        stats.reverse()

        x = self.res(x)
        new_stats = []
        for stat, net in zip(stats, self.stat_convs):
            stat_x = torch.cat(stat, dim=1)  # concatenate mean and std
            new_stats.append(net(stat_x).chunk(2, 1))
        i = 0
        for block in self.up_model:
            if isinstance(block, Conv2dBlock):
                beta, gamma = new_stats[i]
                x = block(x, beta, gamma)
                i += 1
        out = self.cov(x)
        return out

class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std

class MS(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MS, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, inputdim, outputdim, operation, use_bias, use_relu=True, relu=True, ms=False, pono=False, norm=True, front=False, normtype=nn.BatchNorm2d, kerneltype=3):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.front = front
        # initialize normalization
        self.use_relu = use_relu
        self.norm = norm
        self.norm_flag = norm
        if operation == 'conv':
            if kerneltype == 3:
                self.conv = nn.Conv2d(inputdim, outputdim, kernel_size=3, stride=2, padding=1, bias=use_bias)
            elif kerneltype == 5:
                self.conv = nn.Conv2d(inputdim, outputdim, kernel_size=5, stride=2, padding=2, bias=use_bias)
            if normtype==nn.GroupNorm:
                self.norm = normtype(32, outputdim)
            else:
                self.norm = normtype(outputdim)
        elif operation == 'deconv':
            if kerneltype == 3:
                self.conv = nn.ConvTranspose2d(inputdim, outputdim, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias)
            elif kerneltype == 5:
                self.conv = nn.ConvTranspose2d(inputdim, outputdim, kernel_size=5, stride=2,
                                               padding=2, output_padding=1, bias=use_bias)
            if normtype==nn.GroupNorm:
                self.norm = normtype(16, outputdim)
            else:
                self.norm = normtype(outputdim)
        elif operation == 'stat_convs':
            if kerneltype == 3:
                self.conv = nn.Conv2d(inputdim, outputdim, 3, 1, 1)
            elif kerneltype == 5:
                self.conv = nn.Conv2d(inputdim, outputdim, 5, 1, 2)
            if normtype==nn.GroupNorm:
                self.norm = normtype(32, outputdim)
            else:
                self.norm = normtype(outputdim)
        # initialize activation
        self.activation = nn.ReLU(inplace=relu)
        # PONO-MS:
        self.pono = PONO(affine=False) if pono else None
        self.ms = MS() if ms else None

    def forward(self, x, beta=None, gamma=None):
        x = self.conv(x)
        mean, std = None, None
        if self.pono:
            x, mean, std = self.pono(x)
        if self.norm and self.norm_flag:
            x = self.norm(x)
        if self.ms:
            x = self.ms(x, beta, gamma)
        if self.use_relu and self.activation:
            x = self.activation(x)
        if mean is None:
            return x
        else:
            return x, mean, std