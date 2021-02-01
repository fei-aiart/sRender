import torch
import torch.nn as nn
import torchvision.models as models

class Perceptualloss(nn.Module):
    def __init__(self,model,):
        super(Perceptualloss,self).__init__()
        self.vgg_model = model.features
        self.use_layer = set(['3', '8', '15', '22', '29'])
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.mse = torch.nn.MSELoss()

    def forward(self, g, s):
        loss = 0
        i = 0
        for name, module in self.vgg_model._modules.items():

            g, s = module(g), module(s)
            if name in self.use_layer:
                i += 1
                loss += self.mse(g, s) * self.weights[i-1]

        return loss


def vgg_load():
    vgg_model = models.vgg16()
    model_dict = torch.load("/home/meimei/.cache/torch/checkpoints/vgg16-397923af.pth")
    vgg_model.load_state_dict(model_dict)
    return vgg_model