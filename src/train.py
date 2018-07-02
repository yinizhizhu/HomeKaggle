# from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from model import home_t
from sklearn.metrics import roc_auc_score
from dataset import readCSV
import time, random
import pandas as pd
import torch.nn.functional as F
from pycrayon import CrayonClient

cc = CrayonClient(hostname="localhost", port=8889)

names = ['training', 'validation', 'testing', 'loss']
exper = []
for name in names:
    cc.remove_experiment(name)
    exper.append(cc.create_experiment(name))

torch.manual_seed(123)
np.random.seed(123)


class trainer_Order:
    def __init__(self, epochs, l, batchSize):
        self.lr = l
        self.nEpochs = epochs
        self.batchSize = batchSize
        self.outName = 'result_{}_{}.txt'.format(l, batchSize)
        out = open(self.outName, 'w')
        out.close()

        self.cuda = True
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        if self.cuda:
            print('*******Cuda!*******')
            torch.cuda.manual_seed(123)

        print('===> Loading datasets')
        self.tst = readCSV().getTVT()
        self.training_data_loader = data.DataLoader(
            dataset=self.tst[0],
            # num_workers=4,
            batch_size=batchSize,
            shuffle=True)
        self.modelN = 'split.pth'

        self.model = home_t()
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.0807, 0.9193]))

        out = open(self.outName, 'a')
        print >> out, self.model
        out.close()

        if self.cuda:
            print('*******Cuda!!!*******')
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=l, weight_decay=1e-6)
        out = open(self.outName, 'a')
        print >> out, self.optimizer.state_dict()
        out.close()

        self.names = ['acc', 'loss', 'roc']

        self.bestTag = 0.0
        self.bestVal = -1.0
        self.ansTst = 0.0

    def test(self, index, epoch):
        self.model.eval()
        acc = 0.0
        num = len(self.tst[index])
        print num
        y_scores = []
        y_true = []
        for i in xrange(num):
            img, target = self.tst[index][i]
            img = Variable(img, volatile=True)
            if index == 1:
                target2 = Variable(target, volatile=True)
            # print target2
            if self.cuda:
                img = img.cuda()
                if index == 1:
                    target2 = target2.cuda()

            ans = self.model(img)

            # print ans.data.cpu().numpy()
            if index == 1:
                y_true.append(target)
            y_scores.append(F.softmax(ans, dim=1))
            # print ans2.data.cpu().numpy()
            # raw_input('Continue?')

            # gt = np.argmax(ans.data.cpu().numpy()[0])
            # y_scores.append(gt)
            # y_true.append(target.numpy()[0])
            # if i < 12:
            #     print target.numpy()[0], ans.data.cpu().numpy()[0], gt
            # if target.numpy()[0] == gt:
            #     acc += 1.0

        y_scores = torch.cat(y_scores, dim=0)
        y_scores = y_scores[:,1].data.cpu().numpy()
        if index == 1:
            y_true = torch.cat(y_true, dim=0)
            y_true = y_true.numpy()

            roc = roc_auc_score(y_true, y_scores)
            acc = ((y_scores > 0.5) == y_true).mean()

            exper[index].add_scalar_value(self.names[0], acc, epoch)
            exper[index].add_scalar_value(self.names[2], roc, epoch)

            out = open(self.outName, 'a')
            print >> out, "===> Epoch {} Complete: {}. ACC: {:.4f}".format(epoch, names[index], acc)
            print >> out, "===> Epoch {} Complete: {}. ROC: {:.4f}".format(epoch, names[index], roc)
            print "===> Epoch {} Complete: {}. ACC: {:.4f}".format(epoch, names[index], acc)
            print "===> Epoch {} Complete: {}. ROC: {:.4f}".format(epoch, names[index], roc)
            out.close()

            if roc > self.bestVal:
                self.bestVal = roc
                self.bestTag = 1

        elif index == 2:
            if self.bestTag:
                self.bestTag = 0
                output = pd.read_csv('data/sample_submission.csv')
                output['TARGET'] = y_scores
                output.to_csv('test_{}_{}.csv'.format(self.lr, self.batchSize), index=False)

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        num = len(self.training_data_loader)
        print num
        step = 0
        for d in self.training_data_loader:
            img, target = d
            img, target = Variable(img), Variable(target)
            img = img.view(img.size(0), -1)
            # print img.size(), target.size()
            target = target.view(target.size(0))

            if self.cuda:
                img = img.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            ans = self.model(img)
            loss = self.criterion(ans, target)
            epoch_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            exper[3].add_scalar_value(self.names[1], loss.data[0], epoch*num+step)
            step += 1
        epoch_loss = epoch_loss / num

        out = open(self.outName, 'a')
        print >> out, "===> Epoch {} Complete: {}. Loss: {:.4f}".format(epoch, names[0], epoch_loss)
        print "===> Epoch {} Complete: {}. Loss: {:.4f}".format(epoch, names[0], epoch_loss)
        out.close()

    def training(self):
        for epoch in range(self.nEpochs):
            print '*** %03d ***' % epoch
            start = time.time()
            self.test(1, epoch)
            if self.bestTag:
                print '     BestVal:',
                self.test(2, epoch)
            else:
                print '     BestVal =', self.bestVal
            end = time.time()
            testT = end-start

            out = open(self.outName, 'a')
            print >> out, 'testing time: {}\n'.format(testT)
            out.close()

            start = time.time()
            self.train(epoch)
            end = time.time()
            trainT = end-start

            out = open(self.outName, 'a')
            print >> out, 'training time: ', trainT
            out.close()

        out = open(self.outName, 'a')
        print >> out, "Final acc: Val({:.4f}), Tst({:.4f})".format(self.bestVal, self.ansTst)
        out.close()


def main():
    print '--------------- Starting Point ----------------'
    # lrs = [1e-4, 1e-3, 1e-2]
    # lrs = [1e-1, 1]
    lrs = [1e-4]
    # batchSizes = [6, 5, 4, 3, 2]
    batchSizes = [256]
    # batchSizes = [3]
    for batchSize in batchSizes:
        for lr in lrs:
            print '     ', batchSize, lr
            for name in names:
                cc.remove_experiment(name)
                exper.append(cc.create_experiment(name))
            t = trainer_Order(1000, lr, batchSize)
            t.training()
            # raw_input('Taking a Photo!')
            # for name in names:
            #     cc.remove_experiment(name)
try:
    print '*'*47
    main()
except KeyboardInterrupt:
    print '---------Stoped early---------'