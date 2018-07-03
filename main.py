import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from my_dataset import LoanDatasetWrapper, loan_dataset
from sklearn.metrics import roc_auc_score
from model import CreditNet
from logger import Logger
import os

import pdb

#######################

epoch = 1500
batch_size = 512

learning_rate = 1e-4  # 1e-5
lamda = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = Logger('./log')
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

models = {
    'app':  CreditNet(
    feature_grouping=loan_dataset.get_feature_grouping('application_train'),
    critical_feats = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],
    model_params = [3, 8, 2, 256, 128, 32]).to(device),

    # 'bureau': CreditNet(
    # feature_grouping=loan_dataset.get_feature_grouping('bureau'),
    # critical_feats = [],
    # model_params = [3, 4, 2, 64, 32, 16]).to(device),
}


optimizer = optim.Adam([
    {'params': models['app'].parameters(), 'lr': learning_rate, 'weight_decay':lamda},
    #{'params': models['bureau'].parameters(), 'lr': learning_rate, 'weight_decay':lamda}
])

#criterion = nn.NLLLoss()   # log + nll.  # Add class weight
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(
    weight=torch.Tensor([0.0807, 0.9193]).to(device))

#######################

def forward_bacth(data, entry_num):
    app_featrues = {k : v.to(device).float() 
        for k, v in data.items() if k not in ['SK_ID_CURR', 'label']}

    app_out = models['app'](app_featrues)

    # bureau_out = []

    # for entry_id in range(entry_num):

    #     bureau_features, is_empty = loan_dataset.query('bureau', 
    #         'SK_ID_CURR', id_value=data['SK_ID_CURR'][entry_id].item())

    #     if is_empty:
    #         bureau_out.append(torch.tensor([[0, 0]]).float().to(device))
    #         continue

    #     bureau_features = {k : torch.from_numpy(v).to(device).float()
    #         for k, v in bureau_features.items()
    #             if k not in ['SK_ID_CURR', 'SK_ID_BUREAU']}

    #     entry_out = models['bureau'](bureau_features)
    #     entry_out = entry_out.sum(dim=0, keepdim=True)
    #     bureau_out.append(entry_out)

    # bureau_out = torch.cat(bureau_out, dim=0)

    # #out = F.softmax(app_out + bureau_out, dim=1)
    # out = app_out + bureau_out

    return app_out


def test(loader):

    for model in models.values():
        model.eval()

    pred = []
    target = []

    with torch.no_grad():
        for data in loader:

            label = data['label'].to(device)

            out = forward_bacth(data, entry_num=label.shape[0])
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

    for model in models.values():
        model.eval()

    pred = []

    with torch.no_grad():
        for data in test_loader:

            entry_num = data['SK_ID_CURR'].shape[0]

            out = forward_bacth(data, entry_num)

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

        for model in models.values():
            model.train()

        for batch_idx, data in enumerate(train_loader):

            optimizer.zero_grad()

            label = data['label'].to(device)
            out = forward_bacth(data, entry_num=label.shape[0])

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            #out_prob = F.softmax(out, dim=1)

            print('Step {}: Loss-{}'.format(steps, loss.item()))
            logger.scalar_summary('loss_per_step', loss.item(), steps)

            steps += 1




print(models)
train()