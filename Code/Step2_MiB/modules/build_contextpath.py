import torch
from torch import nn
from torchvision import models as tmod
import models

class resnet18(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.features = resnet
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

class resnet50(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.features = resnet
        self.conv1 = self.features.mod1.conv1
        self.bn1 = self.features.mod1.bn1
        self.maxpool1 = self.features.mod1.pool1
        self.layer1 = self.features.mod2
        self.layer2 = self.features.mod3
        self.layer3 = self.features.mod4
        self.layer4 = self.features.mod5

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x) # relu inclusa in bn
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

class resnet101(torch.nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.features = resnet
        self.conv1 = self.features.mod1.conv1
        self.bn1 = self.features.mod1.bn1
        self.maxpool1 = self.features.mod1.pool1
        self.layer1 = self.features.mod2
        self.layer2 = self.features.mod3
        self.layer3 = self.features.mod4
        self.layer4 = self.features.mod5

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x) # relu inclusa in bn
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


def build_contextpath(name, pretrained_path=None, norm_act=nn.BatchNorm2d, out_stride=16):
    if name != 'resnet18':
        resnet = models.__dict__[f'net_{name}'](norm_act=norm_act, output_stride=out_stride)
        pretrained_path = f'{pretrained_path}/{name}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pre_dict['state_dict'].items():
            new_name = k[7:] # remove 'module'
            new_state_dict[new_name] = v
        
        del new_state_dict['classifier.fc.weight']
        del new_state_dict['classifier.fc.bias']
        resnet.load_state_dict(new_state_dict)
        del new_state_dict
        del pre_dict # free memory
    else:
        resnet = tmod.resnet18(pretrained=True)

    if name == 'resnet18': return resnet18(resnet)
    elif name == 'resnet50': return resnet50(resnet)
    elif name == 'resnet101': return resnet101(resnet)

if __name__ == '__main__':
    model_18 = build_contextpath('resnet18')
    model_101 = build_contextpath('resnet101')
    x = torch.rand(1, 3, 256, 256)

    y_18 = model_18(x)
    y_101 = model_101(x)