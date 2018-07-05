import torch.nn as nn


class home_t(nn.Module):
    def __init__(self, addBur, addRest):
        super(home_t, self).__init__()
        inSize = 244
        if addBur:
            inSize += 43
        if addRest:
            inSize += 208
        self.lay1 = nn.Sequential(
            nn.Linear(inSize, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.lay2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.lay3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.lay4 = nn.Sequential(
            nn.Linear(128, 2),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.lay1(x)
        x2 = self.lay2(x1)
        x3 = self.lay3(x2)
        x4 = self.lay4(x3)
        return x4
