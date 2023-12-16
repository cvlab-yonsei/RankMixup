'''
Pytorch implementation of wide resnet.

Reference:
[1] S. Zagoruyko and N. Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.
'''

import torch
import torch.nn as nn
import math
import random
import numpy as np
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Wide_ResNet(nn.Module):

    def __init__(self, block, layers, wfactor, num_classes=10, temp=1.0):
        super(Wide_ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*wfactor, layers[0])
        self.layer2 = self._make_layer(block, 32*wfactor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*wfactor, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion*wfactor, num_classes)
        self.temp = temp

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) / self.temp

        return x


class Wide_ResNet_Mixup(nn.Module):

    def __init__(self, block, layers, wfactor, num_classes=10, temp=1.0, 
                 mixup_alpha = 0.1,
                 layer_mix = 0,
                 num_mixup = 1):
        super(Wide_ResNet_Mixup, self).__init__()
        
        #mixup 
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes

        if layer_mix == None:
            self.layer_mix = 0
        else:
            self.layer_mix = layer_mix

        if num_mixup is not None:
            self.num_mixup = num_mixup
       
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*wfactor, layers[0])
        self.layer2 = self._make_layer(block, 32*wfactor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*wfactor, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion*wfactor, num_classes)
        self.temp = temp

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_multimixup(self, x, target):
        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        mixup = []
        lam = []
        target_re = []
        if self.layer_mix == 0:
            for i in range(self.num_mixup): 
                    x_mix, target_reweighted_curr, lam_current = mixup_process(x, target_reweighted, self.mixup_alpha)
                  
                    mixup.append(x_mix)
                    lam.append(lam_current)
                    target_re.append(target_reweighted_curr)
                    
            x = torch.cat(mixup, dim=0)
            target_re = torch.cat(target_re, dim=0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x_mix = self.maxpool(x_mix)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer_mix == 3:
            for i in range(self.num_mixup): 
                    x_mix, target_reweighted_curr, lam_current = mixup_process(x, target_reweighted, self.mixup_alpha)
                   
                    mixup.append(x_mix)
                    lam.append(lam_current)
                    target_re.append(target_reweighted_curr)
                    
            x = torch.cat(mixup, dim=0)
            target_re = torch.cat(target_re, dim=0)      

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) / self.temp

        return x, target_re, lam

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) / self.temp

        return x

class Wide_ResNet_Tiny(nn.Module):

    def __init__(self, block, layers, wfactor, num_classes=10, temp=1.0):
        super(Wide_ResNet_Tiny, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*wfactor, layers[0])
        self.layer2 = self._make_layer(block, 32*wfactor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*wfactor, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64*block.expansion*wfactor*4, num_classes)
        self.temp = temp

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) / self.temp

        return x

class Wide_ResNet_Mixup_Tiny(nn.Module):

    def __init__(self, block, layers, wfactor, num_classes=10, temp=1.0,
                 mixup_alpha = 0.1,
                 layer_mix = 5,
                 num_mixup = 1):
        super(Wide_ResNet_Mixup_Tiny, self).__init__()
        
        #mixup 
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes

        if layer_mix == None:
            self.layer_mix = 0
        else:
            self.layer_mix = layer_mix

        if num_mixup is not None:
            self.num_mixup = num_mixup        
        
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*wfactor, layers[0])
        self.layer2 = self._make_layer(block, 32*wfactor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*wfactor, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64*block.expansion*wfactor*4, num_classes)
        self.temp = temp

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, int(blocks)):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_multimixup(self, x, target):
        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        mixup = []
        lam = []
        target_re = []
        if self.layer_mix == 0:
            for i in range(self.num_mixup): 
                    x_mix, target_reweighted_curr, lam_current = mixup_process(x, target_reweighted, self.mixup_alpha)
                    mixup.append(x_mix)
                    lam.append(lam_current)
                    target_re.append(target_reweighted_curr)
            x = torch.cat(mixup, dim=0)
            target_re = torch.cat(target_re, dim=0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer_mix == 3:
            for i in range(self.num_mixup): 
                    x_mix, target_reweighted_curr, lam_current = mixup_process(x, target_reweighted, self.mixup_alpha)
                   
                    mixup.append(x_mix)
                    lam.append(lam_current)
                    target_re.append(target_reweighted_curr)
                    
            x = torch.cat(mixup, dim=0)
            target_re = torch.cat(target_re, dim=0) 
             
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) / self.temp
       
        return x, target_re, lam

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) / self.temp

        return x



def wide_resnet_cifar(temp=1.0, num_classes=10, depth=26, width=10, **kwargs):
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    return Wide_ResNet(BasicBlock, [n, n, n], width, num_classes=num_classes, temp=temp, **kwargs)

def wide_resnet_tiny(temp=1.0, num_classes=200, depth=26, width=10, **kwargs):
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    return Wide_ResNet_Tiny(BasicBlock, [n, n, n], width, num_classes=num_classes, temp=temp, **kwargs)

def wide_resnet_cifar_mixup(temp=1.0, num_classes=10, depth=26, width=10, **kwargs):
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    return Wide_ResNet_Mixup(BasicBlock, [n, n, n], width, num_classes=num_classes, temp=temp, **kwargs)

def wide_resnet_tiny_mixup(temp=1.0, num_classes=200, depth=26, width=10, **kwargs):
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    return Wide_ResNet_Mixup_Tiny(BasicBlock, [n, n, n], width, num_classes=num_classes, temp=temp, **kwargs)


def mixup_process(out, target_reweighted, alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted, lam


def to_one_hot(inp, num_classes):
    
    y_onehot = torch.zeros(inp.size(0), num_classes, device=inp.device, requires_grad=False)
    # y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)

    return y_onehot