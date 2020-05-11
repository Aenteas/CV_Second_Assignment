import torch
import torch.nn as nn
import torchvision

class Model(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.num_classes = 7

        if name == 'vgg4_0_2':
            self.model = self.vgg4_0_2()
        elif name == 'vgg4_2_2':
            self.model = self.vgg4_2_2()
        # elif name == 'vgg4_4_2':
        #     self.model = self.vgg4_0_2()
        # elif name == 'vgg4_':
        #     self.model = self.vgg4_0_2()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        # flatten feature map for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_vgg11(self):
        vgg11 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        children = list(vgg11.children())
        # first 4 convolutional layers
        return list(children[0])[:14]

    def vgg4_0_2(self):
        # First 4 convolutional layers of VGG11, average poling and classifier with 2 layers 
        layers = self.extract_vgg11() + [nn.AdaptiveAvgPool2d((7, 7))]
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(nn.Linear(256 * 7 * 7, 512), 
                                        nn.ReLU(True),
                                        nn.Dropout(), 
                                        nn.Linear(512, 7))


    def vgg4_2_2(self):
        # First 4 convolutional layers of VGG11, 2 convolutional layers and classifier with 2 layers
        # Feature map is spatially squeezed to 3x3 shape using max poolings
        layers = self.extract_vgg11() + [nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(True),
                                         nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(True)]
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(nn.Linear(1024 * 7 * 7, 512), 
                                        nn.ReLU(True),
                                        nn.Dropout(), 
                                        nn.Linear(512, 7))