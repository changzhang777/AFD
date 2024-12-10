#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:46:26 2020

@author: pc-3
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):

        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
         
    def forward(self, x):
        x = x.mm(self.w)
        return x 
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #print(out.shape)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, input_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*((input_size // 32) ** 2), num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.T_revision = nn.Linear(num_classes, num_classes, False)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, prejection=False):
        # correction = self.T_revision.weight

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out_1 = out.view(out.size(0), -1)
        out_2 = self.linear(out_1)
        if prejection == True:
            return  out_2, out_1
        else:
            return  out_2




    def feature_forward(self, x, prejection=False):
        correction = self.T_revision.weight
        output = []

        out = F.relu(self.bn1(self.conv1(x)))
        output.append(out)

        out = self.layer1(out)
        output.append(out)

        out = self.layer2(out)
        output.append(out)

        out = self.layer3(out)
        output.append(out)

        out = self.layer4(out)
        # output.append(out)

        out = self.avgpool(out)

        out_1 = out.view(out.size(0), -1)
        output.append(out_1)

        out_2 = self.linear(out_1)

        output.append(out_2)

        output.append(F.softmax(out_2, -1))

        if prejection == True:
            return  out_2, output
        else:
            return  out_2



class ResNet_C_L(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_C_L, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv2 = nn.Conv2d(512*block.expansion, num_classes, kernel_size=4, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(num_classes, num_classes)
        self.T_revision = nn.Linear(num_classes, num_classes, False)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, prejection=False):
        correction = self.T_revision.weight

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        out_1 = out.view(out.size(0), -1)
        out_2 = self.fc(out_1)
        if prejection == True:
            return  out_2, out_1
        else:
            return  out_2



class ResNet_C(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_C, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv2 = nn.Conv2d(512*block.expansion, num_classes, kernel_size=4, stride=1, padding=0, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        out_2 = out.view(out.size(0), -1)
        return  out_2




class ResNet_FS(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_FS, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 512, bias=False)
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.linear1.weight = torch.nn.parameter.Parameter(torch.eye(512))
        self.linear2.weight = torch.nn.parameter.Parameter(torch.eye(512))




    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, prejection=False, logit=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out_1 = out.view(out.size(0), -1)
        out = self.linear1(out_1)
        if self.training or logit:
            logit_1 = self.linear(out)
            out2 = self.linear2(out_1.detach())#
            logit_2 = self.linear(out2)
            out_2 = (logit_1, logit_2)
            out = (out, out2)
        else:
            out_2 = self.linear(out)
        if prejection == True:
            return  out_2, out
        else:
            return  out_2









class ResNet_F(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_F, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.T_revision = nn.Linear(num_classes, num_classes, False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, prejection=False):

        correction = self.T_revision.weight

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out_1 = out.view(out.size(0), -1)
        out_2 = self.linear(out_1)
        if prejection == True:
            return  out_2, out_1
        else:
            return  out_2
        









def ResNet18_C_L(num_classes):
    return ResNet_C_L(BasicBlock, [2,2,2,2], num_classes)

def ResNet18_C(num_classes):
    return ResNet_C(BasicBlock, [2,2,2,2], num_classes)

def ResNet18_FS(num_classes):
    return ResNet_FS(BasicBlock, [2,2,2,2], num_classes)

        
def ResNet18(num_classes, input_size=32):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, input_size)

def ResNet18_F(num_classes):
    return ResNet_F(BasicBlock, [2,2,2,2], num_classes)
    
def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)