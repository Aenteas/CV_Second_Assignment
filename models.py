import torch
import torch.nn as nn
import torchvision

class Model(nn.Module):

    def __init__(self, name):
        super().__init__()
        # 7 emotions
        self.num_classes = 7
        # init model
        if name == 'vgg2_0_2':
            self.model = self.vgg2_0_2()
        elif name == 'vgg2_2_2':
            self.model = self.vgg2_2_2()
        elif name == 'vgg2_4_ilrb_2':
            self.model = self.vgg2_4_ilrb_2()
        elif name == 'vgg2_4_2_conv3':
            self.model = self.vgg2_4_2_conv3()
        elif name == 'vgg2_2_2_conv5':
            self.model = self.vgg2_4_2_conv5()
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        # flatten feature map for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def create_classifier(self, inp):
        return nn.Sequential(nn.Linear(inp, 512), 
                             nn.PReLU(),
                             nn.Dropout(), 
                             nn.Linear(512, 7))

    def extract_vgg11(self):
        vgg11 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        children = list(vgg11.children())
        # first 4 convolutional layers
        return list(children[0])[:7]

    def vgg2_0_2(self):
        # First 4 convolutional layers of VGG11, average poling and classifier with 2 layers 
        layers = self.extract_vgg11() + [nn.AdaptiveAvgPool2d((6, 6))]
        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(128 * 6 * 6)

    def vgg2_2_2(self):
        # First 4 convolutional layers of VGG11, 2 convolutional layers (kernel size 3) and classifier with 2 layers
        layers = self.extract_vgg11() + convBNRelu(128,256,3) + [nn.MaxPool2d(kernel_size=2, stride=2)] + convBNRelu(256,512,3) + [nn.AvgPool2d(kernel_size=2, stride=2)]

        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(512 * 6 * 6)

    def vgg2_4_2_conv5(self):
        # First 4 convolutional layers of VGG11, 2 convolutional layers (kernel size 5) and classifier with 2 layers
        layers = self.extract_vgg11() + [*convBNRelu(128,128,5, padding=0),
                                         *convBNRelu(128,128,5, padding=0),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         *convBNRelu(128,256,5),
                                         *convBNRelu(256,512,3)]

        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(512 * 6 * 6)

    def vgg2_4_2_conv3(self):
        # First 4 convolutional layers of VGG11, 2 convolutional layers (kernel size 3) and classifier with 2 layers
        # Feature map is spatially squeezed to 3x3 shape by max poolings
        layers = self.extract_vgg11() + [*convBNRelu(128,128,3),
                                         *convBNRelu(128,128,3),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         *convBNRelu(128,256,3),
                                         *convBNRelu(256,512,3),
                                         nn.AvgPool2d(kernel_size=2, stride=2),]

        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(512 * 6 * 6)

    def vgg2_4_ilrb_2(self):
        # First 4 convolutional layers of VGG11, 2 inverted residual blocks separated by average pooling and classifier with 2 layers
        layers = self.extract_vgg11() + [inverted_linear_residual_block(128,1024,128),
                                         inverted_linear_residual_block(128,1024,128),
                                         nn.MaxPool2d(kernel_size=2, stride=2),
                                         inverted_linear_residual_block(128,1280,256),
                                         inverted_linear_residual_block(256,1280,512),
                                         nn.AvgPool2d(kernel_size=2, stride=2),]
        self.features = nn.Sequential(*layers)
        self.classifier = self.create_classifier(512 * 6 * 6)

class inverted_linear_residual_block(nn.Module):
    # implementation of basic building block of MobilnetV2
    def __init__(self, inp, exp, out):
        super(inverted_linear_residual_block, self).__init__()
        layers = convBNRelu(inp,exp,1, padding=0) + convBNRelu(exp,exp,3,exp) + [nn.Conv2d(exp,out,1, padding = 0, bias=False), nn.BatchNorm2d(out)]
        self.layers = nn.Sequential(*layers)
        self.skip_connect = inp == out
    def forward(self, x):
        # skip connection
        if self.skip_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

def convBNRelu(inp, out, kernel_size, groups=1, padding=1):
    # when groups = inp we have depthwise convolution
    return [nn.Conv2d(inp, out, kernel_size=kernel_size, padding=padding, bias=False, groups=groups), nn.BatchNorm2d(out), nn.PReLU()]
