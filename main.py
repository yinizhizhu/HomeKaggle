import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from my_dataset import LoanDatasetWrapper
from sklearn.metrics import roc_auc_score
from model import CreditNet
from logger import Logger
import os

import pdb

#######################

epoch = 3
batch_size = 256

learning_rate = 1e-3  # 1e-5
lamda = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = Logger('.')
#######################

train_dataset = LoanDatasetWrapper(mode='train')
val_dataset = LoanDatasetWrapper(mode='val')
test_dataset = LoanDatasetWrapper(mode='test')

train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)


print(len(train_dataset))
#######################

model = CreditNet(train_dataset.get_feature_grouping()).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamda)

#criterion = nn.NLLLoss()   # log + nll.  # Add class weight
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(
    weight=torch.Tensor([0.0807, 0.9193]).to(device))

#######################


def test(loader):

    model.eval()

    pred = []
    target = []

    with torch.no_grad():
        for data in loader:

            featrues = {k : v.to(device).float() 
                    for k, v in data.items() if k not in ['id', 'label']}
            label = data['label'].to(device)

            out = model(featrues)
            out_prob = F.softmax(out, dim=1)

            target.append(label)
            pred.append(out_prob)

    target = torch.cat(target, dim=0)
    pred = torch.cat(pred, dim=0)

    pred = pred[:,1].cpu().numpy()
    target = target.cpu().numpy()

    auc = roc_auc_score(target, pred)
    acc = ((pred > 0.5) == target).mean()

    return auc, acc, pred


def save_result():

    model.eval()

    pred = []

    with torch.no_grad():
        for data in test_loader:
            featrues = {k : v.to(device).float() 
                    for k, v in data.items() if k not in ['id', 'label']}
            out = model(featrues)
            out_prob = F.softmax(out, dim=1)
            pred.append(out_prob)

    pred = torch.cat(pred, dim=0)
    pred = pred[:,1].cpu().numpy()

    sample_file = os.path.join('../input', 'sample_submission.csv')
    result = pd.read_csv(sample_file)
    result['TARGET'] = pred

    result.to_csv('./test_result.csv', index=False)


def train():

    steps = 0

    for e in range(epoch):

        if e % 1 == 0:

            save_result()

            val_auc, val_acc, _ = test(val_loader)
            train_auc, train_acc, _ = test(train_loader)

            print('Epoch {}: val_auc-{}'.format(e+1, val_auc))
            print('Epoch {}: val_acc-{}'.format(e+1, val_acc))
            print('Epoch {}: train_auc-{}'.format(e+1, train_auc))
            print('Epoch {}: train_acc-{}'.format(e+1, train_acc))

            logger.scalar_summary('val_auc', val_auc, e+1)
            logger.scalar_summary('val_acc', val_acc, e+1)
            logger.scalar_summary('train_auc', train_auc, e+1)
            logger.scalar_summary('train_acc', train_acc, e+1)

        model.train()

        for batch_idx, data in enumerate(train_loader):

            featrues = {k : v.to(device).float() 
                for k, v in data.items() if k not in ['id', 'label']}
            label = data['label'].to(device)

            optimizer.zero_grad()
            out = model(featrues)

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            #out_prob = F.softmax(out, dim=1)

            print('Step {}: Loss-{}'.format(steps, loss.item()))
            logger.scalar_summary('loss_per_step', loss.item(), steps)

            steps += 1




print(model)
train()