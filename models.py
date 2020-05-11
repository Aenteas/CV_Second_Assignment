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
        elif name == 'vgg4_2_ilrb_2':
            self.model = self.vgg4_2_ilrb_2()
        elif name == 'vgg4_4_ilrb_2':
            self.model = self.vgg4_4_ilrb_2()
        elif name == 'vgg4_2_2_conv5':
            self.model = self.vgg4_2_2_conv5()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        # flatten feature map for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def convBNRelu(self, inp, out, kernel_size, groups=1):
        # when groups = inp we have depthwise convolution
        return [nn.Conv2d(inp, out, kernel_size=kernel_size, padding=1, bias=False, groups=groups), nn.BatchNorm2d(out), nn.ReLU(True)]

    def create_classifier(self, inp):
        return nn.Sequential(nn.Linear(inp, 512), 
                             nn.ReLU(True),
                             nn.Dropout(), 
                             nn.Linear(512, 7))

    def extract_vgg11(self):
        vgg11 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        children = list(vgg11.children())
        # first 4 convolutional layers
        return list(children[0])[:14]

    def vgg4_0_2(self):
        # First 4 convolutional layers of VGG11, average poling and classifier with 2 layers 
        layers = self.extract_vgg11() + [nn.AdaptiveAvgPool2d((7, 7))]
        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(256 * 7 * 7)

    def vgg4_2_2(self):
        # First 4 convolutional layers of VGG11, 2 convolutional layers and classifier with 2 layers
        # Feature map is spatially squeezed to 3x3 shape using max poolings
        layers = self.extract_vgg11() + [nn.MaxPool2d(kernel_size=2, stride=2),
                                         self.convBNRelu(256,512,3),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         self.convBNRelu(512,1024,3)]

        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(1024 * 3 * 3)

    def vgg4_2_2_conv5(self):
        layers = self.extract_vgg11() + [nn.MaxPool2d(kernel_size=2, stride=2),
                                         self.convBNRelu(256,512,5),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         self.convBNRelu(512,1024,3)]

        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(1024 * 4 * 4)

    def vgg4_2_ilrb_2(self):
        layers = self.extract_vgg11() + [inverted_linear_residual_block(256,1024,512),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         inverted_linear_residual_block(512,1024,512), 
                                         nn.AdaptiveAvgPool2d((3, 3))]
        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(512 * 3 * 3)

    def vgg4_4_ilrb_2(self):
        layers = self.extract_vgg11() + [inverted_linear_residual_block(256,512,256),
                                         inverted_linear_residual_block(256,1024,512),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         inverted_linear_residual_block(512,1024,256),
                                         inverted_linear_residual_block(512,1024,512),
                                         nn.AdaptiveAvgPool2d((3, 3))]
        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(512 * 3 * 3)

class inverted_linear_residual_block(nn.Module):
    def __init__(self, inp, exp, out):
        super(inverted_linear_residual_block, self).__init__()
        layers = [self.convBNRelu(inp,exp,1), self.convBNRelu(exp,exp,3,exp), nn.Conv2d(exp,out,1, bias=False), nn.BatchNorm2d(out)]
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        # skip connection
        return x + self.layers(x)
