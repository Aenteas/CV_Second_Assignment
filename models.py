import torch
import torch.nn as nn
import torchvision

class Model(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.num_classes = 7

        if name == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)
        elif name == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
        elif name == 'resnet101':
            self.model = torchvision.models.resnet101(pretrained=True)
        else:
            raise NotImplementedError

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    def vgg4_0_2():


    def vgg4_2_2():

    def vgg4_4_2():

    def vgg4_