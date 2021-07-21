import torch
from torch import nn
from modules.build_contextpath import build_contextpath
import warnings
warnings.filterwarnings(action='ignore')

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, norm_act=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_act(out_channels)
        # self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        # return self.relu(self.bn(x))
        return self.bn(x) # ReLU included in BN

class Spatial_path(torch.nn.Module):
    def __init__(self, norm_act=nn.BatchNorm2d):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64, norm_act=norm_act)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128, norm_act=norm_act)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256, norm_act=norm_act)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        # x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm_act=nn.BatchNorm2d):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels
        # fixed number of out channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=out_channels, stride=1, norm_act=norm_act)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(torch.nn.Module):
    def __init__(self, context_path='resnet101', pretrained_path=None, out_channels=128, norm_act=nn.BatchNorm2d, out_stride=16, is_old=False):
        super().__init__()
        self.is_old = is_old

        # build spatial path
        self.spatial_path = Spatial_path(norm_act=norm_act)

        # build context path
        self.context_path = build_contextpath(name=context_path, pretrained_path=pretrained_path, norm_act=norm_act)

        if context_path == 'resnet101':
            # build attention refinement module  for resnet 101
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(3328, out_channels, norm_act=norm_act)
        
        elif context_path == 'resnet50':
            # build attention refinement module  for resnet 50
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(3328, out_channels, norm_act=norm_act)

        elif context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(1024, out_channels, norm_act=norm_act)
        else:
            print('Error: unspport context_path network \n')

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.spatial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.feature_fusion_module)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, phase='train'):
        # output of spatial path
        sx = self.spatial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear') after final convolution in segmentation_module.py
        # result = self.conv(result) final convolution in segmentation_module.py
        
        if not self.is_old:
            if self.training == True:
                return result, cx1, cx2
            return (result,)
        else:
            if phase == 'train': # train of model, not of model_old
                return result, cx1, cx2
            return (result,)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = BiSeNet(32, 'resnet18')
    # model = nn.DataParallel(model)

    model = model.cuda()
    x = torch.rand(2, 3, 256, 256)
    record = model.parameters()
    # for key, params in model.named_parameters():
    #     if 'bn' in key:
    #         params.requires_grad = False
    from utils import group_weight
    # params_list = []
    # for module in model.mul_lr:
    #     params_list = group_weight(params_list, module, nn.BatchNorm2d, 10)
    # params_list = group_weight(params_list, model.context_path, torch.nn.BatchNorm2d, 1)

    print(model.parameters())