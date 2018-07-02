import torch.utils.data as data
import torch, random, csv
import numpy as np


def checkAlpha(str):
    for i in str:
        if i >= 'a' and i < 'e':
            return True
        elif i > 'e' and i < 'z':
            return True
        elif i >= 'A' and i <= 'Z':
            return True
    return False


def oneHot(n, index):
    n -= 1
    index -= 1
    ans = [0.0 for i in xrange(n)]
    if index >= 0:
        ans[index] = 1.0
    else:
        ans[0] = -1.0
    return ans


class dataVal(data.Dataset):
    def __init__(self, data):
        super(dataVal, self).__init__()
        self.datas = data

    def __getitem__(self, item):
        data, target = self.datas[item]
        data = np.array([data])
        target = np.array([target])
        # print len(data)
        data = torch.FloatTensor(data)
        target = torch.LongTensor(target)
        return data, target

    def __len__(self):
        return len(self.datas)


class readCSV(data.Dataset):
    def __init__(self):
        super(readCSV, self).__init__()
        self.trn = []
        self.val = []
        self.tst = []

    def loadSplit(self):
        all = []
        all_p = 0.0
        attrs = []
        types = [None for i in xrange(122)]
        dicts = [dict() for i in xrange(122)]
        tags = [1 for i in xrange(122)]
        maxN = [0.0 for i in xrange(122)]
        with open('data/application_train.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                feature = []
                label = None
                if counter:
                    for i in xrange(122):
                        if i == 1:
                            label = int(row[i])
                            if label == 1:
                                all_p += 1.0
                        elif i > 1:
                            if len(row[i]) == 0:
                                feature.append(-1.0)
                            elif checkAlpha(row[i]):
                                if types[i] is not None and types[i] == 'num':
                                    print row[i]
                                types[i] = 'str'
                                if row[i] not in dicts[i]:
                                    dicts[i][row[i]] = tags[i]
                                    tags[i] += 1
                                feature.append(dicts[i][row[i]])
                            else:
                                if types[i] is None:
                                    types[i] = 'num'
                                tmp = abs(float(row[i]))
                                maxN[i] = max(maxN[i], tmp)
                                feature.append(tmp)
                    all.append([feature, label])
                else:
                    for i in xrange(2, 122):
                        attrs.append(row[i])
                counter += 1
        counter -= 1
        index = [i for i in xrange(counter)]
        random.shuffle(index)

        output = open('train.csv', 'w')
        print >> output, '      Train:'
        for i in xrange(10):
            print >> output, i, all[i][1], all[i][0]
        print 'Positive rate:', all_p/counter, '({})'.format(all_p), ', Negative rate:', (counter - all_p)/counter

        for i in xrange(122):
            if types[i] == 'str':
                print i, tags[i]-1, attrs[i], dicts[i]

        val_p = 0.0
        val_len = int(counter*0.10)
        lens = []
        for i in xrange(val_len):
            one_hot = []
            feature, label = all[index[i]]
            for k in xrange(2, 122):
                if types[k] == 'str':
                    # print tags[k], feature[k-2]
                    one_hot += oneHot(tags[k], feature[k-2])
                else:
                    tmp = feature[k-2]
                    if tmp > 0:
                        one_hot.append(tmp/maxN[k])
                    else:
                        one_hot.append(tmp)
            self.val.append([one_hot, label])
            lens.append(len(one_hot))
            if all[index[i]][1] == 1:
                val_p += 1.0
        print 'maxLen:', max(lens), min(lens)

        lens = []
        for i in xrange(val_len, counter):
            one_hot = []
            feature, label = all[index[i]]
            for j in xrange(120):
                k = j + 2
                if types[k] == 'str':
                    # print tags[k], feature[k-2]
                    one_hot += oneHot(tags[k], feature[k-2])
                else:
                    tmp = feature[k-2]
                    if tmp > 0:
                        one_hot.append(tmp/maxN[k])
                    else:
                        one_hot.append(tmp)
            lens.append(len(one_hot))
            self.trn.append([one_hot, label])
        print 'maxLen:', max(lens), min(lens)
        print 'one_hot:', len(one_hot), one_hot
        print 'Train { Positive rate:', (all_p-val_p)/len(self.trn), '}'
        print 'Validate { Positive rate:', val_p/val_len, '}'
        print val_len, len(self.val), counter-val_len, len(self.trn), counter, len(all)

        lens = []
        with open('data/application_test.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    feature = []
                    for j in xrange(122):
                        if j <= 1:
                            continue
                        i = j-1
                        if types[j] == 'str':
                            if len(row[i]) == 0:
                                feature += oneHot(tags[j], -1)
                            else:
                                if row[i] not in dicts[j]:
                                    print row[i], dicts[j]
                                feature += oneHot(tags[j], dicts[j][row[i]])
                        else:
                            if len(row[i]) == 0:
                                feature.append(-1.0)
                            else:
                                feature.append(abs(float(row[i]))/maxN[j])
                    self.tst.append([feature, random.randint(0,1)])
                    lens.append(len(feature))
                counter += 1
        counter -= 1
        print counter, len(self.tst)
        print 'maxLen:', max(lens), min(lens)
        print len(feature), feature

        print >> output, '      Test:'
        for i in xrange(10):
            print >> output, i, self.tst[i][1], self.tst[i][0]
        output.close()

    def getTVT(self):
        self.loadSplit()
        print len(self.trn), len(self.val), len(self.tst)
        return [dataVal(self.trn), dataVal(self.val), dataVal(self.tst)]


# d = readCSV()
# d.loadSplit()