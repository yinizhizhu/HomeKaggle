import torch
import torch.nn as nn
from torch.nn.init import *
import torchvision


class home(nn.Module):
    def __init__(self):
        super(home, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Linear(120, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.lay2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.lay3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.lay4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.lay5 = nn.Sequential(
            nn.Linear(256, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
        )
        self.lay6 = nn.Sequential(
            nn.Linear(120, 2),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.lay1(x)
        x2 = self.lay2(x1)
        x3 = self.lay3(x2)
        x_m = self.mid(x3)
        # x4 = self.lay4(x2+x_m)
        # x5 = self.lay5(x1+x4)
        # x6 = self.lay6(x+x5)
        x4 = self.lay4(x_m)
        x5 = self.lay5(x4)
        x6 = self.lay6(x5)
        # print x6
        # raw_input('')
        return x6


class home_s(nn.Module):
    def __init__(self):
        super(home_s, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Linear(120, 256),
            nn.ReLU(inplace=True),
        )
        self.lay2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )
        self.lay4 = nn.Sequential(
            nn.Linear(512, 2),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.lay1(x)
        x2 = self.lay2(x1)
        x4 = self.lay4(x2)
        return x4


# d = torchvision.models.resnet18(pretrained=True)
# print d