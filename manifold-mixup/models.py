# Referenced: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)


class IdentityMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityMapping, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        out = x
        # option A from https://arxiv.org/pdf/1512.03385.pdf
        if self.in_channels != self.out_channels:
            num_of_pad = self.out_channels - self.in_channels
            out = F.pad(out[:, :, ::2, ::2], (0, 0, 0, 0, num_of_pad // 2, num_of_pad // 2))

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = IdentityMapping(in_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(x)
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, layers, num_classes=10):
        super(Resnet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(layers[0], 16, 16, stride=1)
        self.layers2 = self._make_layer(layers[1], 16, 32, stride=2)
        self.layers3 = self._make_layer(layers[2], 32, 64, stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        # weight initialization
        self.apply(init_weights)

    def _make_layer(self, num_of_blocks, in_channels, out_channels, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for i in range(num_of_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avg_pool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out
