# from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from home_model import home, home_s
from home_dataset import readCSV
import time, random
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
    def __init__(self, epochs, l):
        self.nEpochs = epochs
        self.outName = 'result.txt'
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
        self.modelN = 'split.pth'

        self.model = home()
        # self.model = home_s()
        self.criterion = nn.CrossEntropyLoss()

        if self.cuda:
            print('*******Cuda!!!*******')
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=l)

        self.names = ['ACC', 'loss']

        self.bestTag = 0.0
        self.bestVal = 1.0
        self.ansTst = 0.0
        self.loadClass()

    def loadClass(self):
        self.filenames = []
        input = open('sample_submission.csv', 'r')
        input.readline()
        for line in input.readlines():
            self.filenames.append(line.strip().split(',')[0])
        input.close()

    def test(self, index, epoch):
        if self.bestTag:
            output = open('test.csv', 'w')
            print >> output, 'SK_ID_CURR,TARGET'
        testLoss = 0
        acc = 0.0
        self.tst[index]
        num = len(self.tst[index])
        print num
        for i in xrange(num):
            img, target = self.tst[index][i]
            # if i < 12:
            #     print img.numpy()[0]
            img, target2 = Variable(img, volatile=True), Variable(target, volatile=True)
            # print target2
            if self.cuda:
                img = img.cuda()
                target2 = target2.cuda()

            ans = self.model(img)
            loss = self.criterion(ans, target2)
            testLoss += loss.data[0]

            gt = np.argmax(ans.data.cpu().numpy()[0])
            if i < 12:
                print target.numpy()[0], ans.data.cpu().numpy()[0], gt
            if self.bestTag:
                print >> output, '{},{:.1f}'.format(self.filenames[i], gt)
            if target.numpy()[0] == gt:
                acc += 1.0
        acc /= num
        testLoss = testLoss / num
        if index == 1:
            if testLoss < self.bestVal:
                self.bestVal = testLoss
                self.bestTag = 1
        elif index == 2:
            if self.bestTag:
                self.ansTst = testLoss
                self.bestTag = 0
                output.close()
        exper[index].add_scalar_value(self.names[0], acc, epoch)
        exper[index].add_scalar_value(self.names[1], testLoss, epoch)

        out = open(self.outName, 'a')
        print >> out, "===> Epoch {} Complete: {}. Loss: {:.4f}".format(epoch, names[index], testLoss)
        print >> out, "===> Epoch {} Complete: {}. ACC: {:.4f}".format(epoch, names[index], acc)
        print "===> Epoch {} Complete: {}. Loss: {:.4f}".format(epoch, names[index], testLoss)
        print "===> Epoch {} Complete: {}. ACC: {:.4f}".format(epoch, names[index], acc)
        out.close()

    def train(self, epoch):
        epoch_loss = 0
        num = len(self.tst[0])
        print num
        for i in xrange(num):
            img, target = self.tst[0][i]
            img, target = Variable(img), Variable(target)
            # print target
            # print img.size(), target.size()

            # img = img.view(img.size(0), -1)
            # print img.size(), target.size()
            # target = target.view(target.size(0))
            # print target

            if self.cuda:
                img = img.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            ans = self.model(img)
            loss = self.criterion(ans, target)
            epoch_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
        epoch_loss = epoch_loss / num

        exper[3].add_scalar_value(self.names[1], epoch_loss, epoch)

        out = open(self.outName, 'a')
        print >> out, "===> Epoch {} Complete: {}. Loss: {:.4f}".format(epoch, names[0], epoch_loss)
        print "===> Epoch {} Complete: {}. Loss: {:.4f}".format(epoch, names[0], epoch_loss)
        out.close()

    def training(self):
        for epoch in range(self.nEpochs):
            print '*** %03d ***' % epoch
            start = time.time()
            self.model.train()
            self.train(epoch)
            end = time.time()
            trainT = end-start

            out = open(self.outName, 'a')
            print >> out, 'training time: ', trainT
            out.close()

            start = time.time()
            self.model.eval()
            # self.test(0, epoch)
            self.test(1, epoch)
            self.test(2, epoch)
            end = time.time()
            testT = end-start

            out = open(self.outName, 'a')
            print >> out, 'testing time: {}\n'.format(testT)
            out.close()

        out = open(self.outName, 'a')
        print >> out, "Final acc: Val({:.4f}), Tst({:.4f})".format(self.bestVal, self.ansTst)
        out.close()


def main():
    print '--------------- Starting Point ----------------'
    t = trainer_Order(15, 1e-2)
    t.training()
    raw_input('Taking a Photo!')
    for name in names:
        cc.remove_experiment(name)
try:
    print '*'*47
    main()
except KeyboardInterrupt:
    print '---------Stoped early---------'