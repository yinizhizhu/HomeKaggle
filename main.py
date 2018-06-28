import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from my_dataset import LoanDatasetWrapper
from model import CreditNet
import os

import pdb

#######################

epoch = 50
batch_size = 128

learning_rate = 1e-3
lamda = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################

train_dataset = LoanDatasetWrapper(mode='train')
train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

#######################

model = CreditNet(train_dataset.get_feature_grouping())

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamda)

#criterion = nn.NLLLoss()   # log + nll.  # Add class weight
criterion = nn.CrossEntropyLoss()

#######################

steps = 0

for e in range(epoch):

    model.train()

    for batch_idx, data in enumerate(train_loader):

        featrues = {k : v.to(device).float() 
            for k, v in data.items() if k != 'label'}

        label = data['label'].to(device)

        optimizer.zero_grad()
        out = model(featrues)

        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        #out_prob = F.softmax(out, dim=1)

        print('Step {}: Loss-{}'.format(steps, loss.item()))
        #logger.scalar_summary('loss_per_step', loss.item(), steps)

        steps += 1
