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


class dataTrn(data.Dataset):
    def __init__(self, data):
        super(dataTrn, self).__init__()
        self.dataP = []
        self.dataN = []
        for i in data:
            if i[1] == 1:
                self.dataP.append(i)
            else:
                self.dataN.append(i)
        self.numP = len(self.dataP)
        self.numN = len(self.dataN)
        self.lists = [i for i in xrange(self.numN)]

    def __getitem__(self, item):
        i = random.randint(0, self.numP-1)
        data = [self.dataP[i][0]]
        target = [self.dataP[i][1]]
        for i in xrange(2*item, 2*(item+1)):
            data.append(self.dataN[i][0])
            target.append(self.dataN[i][1])

        data = np.array(data)
        target = np.array(target)
        # print len(data)
        data = torch.FloatTensor(data)
        target = torch.LongTensor(target)
        return data, target

    def __len__(self):
        random.shuffle(self.lists)
        return self.numN/2


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
        types = [None for i in xrange(122)]
        dicts = [dict() for i in xrange(122)]
        tags = [1.0 for i in xrange(122)]
        with open('application_train.csv', 'r') as f:
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
                                    tags[i] += 1.0
                                feature.append(dicts[i][row[i]])
                            else:
                                if types[i] is None:
                                    types[i] = 'num'
                                feature.append(float(row[i]))
                    all.append([feature, label])
                counter += 1
        counter -= 1
        index = [i for i in xrange(counter)]
        random.shuffle(index)

        output = open('train.csv', 'w')
        for i in xrange(10):
            print >> output, i, all[i][1], all[i][0]
        print 'Positive rate:', all_p/counter, '({})'.format(all_p), ', Negative rate:', (counter - all_p)/counter

        for i in xrange(122):
            if types[i] == 'str':
                print i, tags[i]-1, dicts[i]

        val_len = int(counter*0.1)
        val_p = 0.0
        for i in xrange(val_len):
            self.val.append(all[index[i]])
            if all[index[i]][1] == 1:
                val_p += 1.0
        for i in xrange(val_len, counter):
            self.trn.append(all[index[i]])
        print 'Train { Positive rate:', (all_p-val_p)/len(self.trn), '}'
        print 'Validate { Positive rate:', val_p/val_len, '}'

        print len(feature), feature

        print val_len, len(self.val), counter-val_len, len(self.trn), counter, len(all)

        with open('application_test.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                feature = []
                if counter:
                    for j in xrange(122):
                        if j <= 1:
                            continue
                        else:
                            i = j-1
                        if len(row[i]) == 0:
                            feature.append(-1.0)
                        elif types[j] == 'str':
                            feature.append(dicts[j][row[i]])
                        else:
                            feature.append(float(row[i]))
                    self.tst.append([feature, 1])
                counter += 1
        counter -= 1
        print counter, len(self.tst)
        print len(feature), feature

        for i in xrange(10):
            print >> output, i, self.tst[i][1], self.tst[i][0]
        output.close()

    def getTVT(self):
        self.loadSplit()
        print len(self.trn), len(self.val), len(self.tst)
        return [dataTrn(self.trn), dataVal(self.val), dataVal(self.tst)]

# d = readCSV()
# d.loadSplit()