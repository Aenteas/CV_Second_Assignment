import torch
import torch.nn as nn
import torchvision

class Model(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.num_classes = 7

        if name == 'vgg4_0_2':
            self.model = self.vgg4_0_2()
        # elif name == 'vgg4_2_2':
        #     self.model = self.vgg4_0_2()
        # elif name == 'vgg4_4_2':
        #     self.model = self.vgg4_0_2()
        # elif name == 'vgg4_':
        #     self.model = self.vgg4_0_2()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_vgg11(self):
        vgg11 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        children = list(vgg11.children())
        # first 4 convolutional layers
        print(list(children[0])[:14])
        return list(children[0])[:14]

    def vgg4_0_2(self):
        layers = self.extract_vgg11() + [nn.AdaptiveAvgPool2d((7, 7))]
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(nn.Linear(256 * 7 * 7, 512), 
                                        nn.ReLU(True),
                                        nn.Dropout(), 
                                        nn.Linear(512, 7))