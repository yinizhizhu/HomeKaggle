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
        # for i in xrange(len(data)):
        #     print i, type(data[i]), data[i]
        # print type(data), data
        # print type(target), target
        data = np.array([data])
        target = np.array([target])
        # print len(data)
        data = torch.FloatTensor(data)
        target = torch.LongTensor(target)
        return data, target

    def __len__(self):
        return len(self.datas)


class readCSV(data.Dataset):
    def __init__(self, addBur, addRest):
        super(readCSV, self).__init__()
        self.trn = []
        self.val = []
        self.tst = []
        self.addBur = addBur
        self.addRest = addRest

    def loadApplication(self):
        if self.addBur:
            self.loadBureau()
        if self.addRest:
            self.loadPos()
        print '*'*15, 'Loading application', '*'*15
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
                                feature.append(0.0)
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
                    all.append([feature, label, row[0]])
                else:
                    for i in xrange(0, 122):
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
            feature, label, sk_id_curr = all[index[i]]
            for k in xrange(2, 122):
                if types[k] == 'str':
                    # print tags[k], feature[k-2]
                    one_hot += oneHot(tags[k], feature[k-2])
                else:
                    one_hot.append(feature[k-2]/maxN[k])
            if self.addBur:
                one_hot += self.cur_balance[sk_id_curr]
            if self.addRest:
                one_hot += self.cur_pos[sk_id_curr]
            self.val.append([one_hot, label])
            lens.append(len(one_hot))
            if all[index[i]][1] == 1:
                val_p += 1.0
        print 'maxLen:', max(lens), min(lens)

        lens = []
        for i in xrange(val_len, counter):
            one_hot = []
            feature, label, sk_id_curr = all[index[i]]
            for j in xrange(120):
                k = j + 2
                if types[k] == 'str':
                    # print tags[k], feature[k-2]
                    one_hot += oneHot(tags[k], feature[k-2])
                else:
                    one_hot.append(feature[k-2]/maxN[k])
            if self.addBur:
                one_hot += self.cur_balance[sk_id_curr]
            if self.addRest:
                one_hot += self.cur_pos[sk_id_curr]
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
                                feature.append(0.0)
                            else:
                                feature.append(abs(float(row[i]))/maxN[j])
                    if self.addBur:
                        feature += self.cur_balance[row[0]]
                    if self.addRest:
                        feature += self.cur_pos[row[0]]
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

    def loadBureau(self):
        # sk_id_curr -> sk_id_bureau (one-to-multi)
        # sk_id_bureau -> balance (one-to-multi)
        print '*'*15, 'Loading bureau', '*'*15
        self.cur_balance = dict()
        with open('data/application_train.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    if row[0] not in self.cur_balance:
                        self.cur_balance[row[0]] = None
                    else:
                        print 'The sk_id_curr is not unique:', row[0]
                counter += 1

        with open('data/application_test.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    if row[0] not in self.cur_balance:
                        self.cur_balance[row[0]] = None
                    else:
                        print 'The sk_id_curr is not unique:', row[0]
                counter += 1

        attrs = []
        types = [None for i in xrange(17)]
        dicts = [dict() for i in xrange(17)]
        tags = [1 for i in xrange(17)]
        maxN = [0.0 for i in xrange(17)]
        self.bur_cur = dict()
        with open('data/bureau.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    if row[1] in self.bur_cur:
                        if row[0] != self.bur_cur[row[1]]:
                            print 'Not one-to-one:', row[0], self.bur_cur[row[1]]
                    else:
                        self.bur_cur[row[1]] = row[0]

                    feature = []
                    for i in xrange(17):
                        if i > 1:
                            if len(row[i]) == 0:
                                feature.append(0.0)
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
                                tmp = float(row[i])
                                maxN[i] = max(maxN[i], abs(tmp))
                                feature.append(tmp)
                    self.cur_balance[row[0]] = feature + [0.0 for i in xrange(8)]
                else:
                    for i in xrange(0, 17):
                        attrs.append(row[i])
                counter += 1
        print 'The length of feature:', len(feature), feature

        for i in xrange(17):
            if types[i] == 'str':
                print tags[i] - 1, attrs[i], dicts[i]

        maps = {'C':7, 'X':8, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6}
        with open('data/bureau_balance.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    if row[0] in self.bur_cur:
                        sk_id_cur = self.bur_cur[row[0]]
                        if sk_id_cur in self.cur_balance:
                            self.cur_balance[sk_id_cur][14+maps[row[2]]] = 1.0
                counter += 1

        lens = []
        counter = 0.0
        for key in self.cur_balance:
            feature = self.cur_balance[key]
            if feature is not None:
                counter += 1.0
                ans = []
                for i in xrange(17):
                    if i > 1:
                        if types[i] == 'str':
                            ans += oneHot(tags[i], feature[i - 2])
                        else:
                            ans.append(feature[i - 2] / maxN[i])
                for i in xrange(17, 25):
                    ans.append(feature[i - 2])
                lens.append(len(ans))
                self.cur_balance[key] = ans
            else:
                self.cur_balance[key] = [0.0 for i in xrange(43)]
        print ans
        print max(lens), min(lens)
        print 'The ratio of bureau with data:', counter/len(self.cur_balance)

    def loadPos(self):
        print '*'*15, 'Loading pos_cash', '*'*15
        self.cur_pos = dict()
        with open('data/application_train.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    if row[0] not in self.cur_pos:
                        self.cur_pos[row[0]] = None
                    else:
                        print 'The sk_id_curr is not unique:', row[0]
                counter += 1

        with open('data/application_test.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    if row[0] not in self.cur_pos:
                        self.cur_pos[row[0]] = None
                    else:
                        print 'The sk_id_curr is not unique:', row[0]
                counter += 1

        # ***************** read the pos_cash_balance.csv ********************************#
        attrs = []
        types = [None for i in xrange(8)]
        dicts = [dict() for i in xrange(8)]
        tags = [1 for i in xrange(8)]
        maxN = [0.0 for i in xrange(8)]
        self.pre_credit = dict()
        with open('data/POS_CASH_balance.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    feature = []
                    for i in xrange(8):
                        if i > 2:
                            if len(row[i]) == 0:
                                feature.append(0.0)
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
                                tmp = float(row[i])
                                maxN[i] = max(maxN[i], abs(tmp))
                                feature.append(tmp)
                    if row[1] in self.cur_pos:
                        if self.cur_pos[row[1]] is None or self.cur_pos[row[1]][1] < float(row[2]):
                            self.cur_pos[row[1]] = [feature, float(row[2])]
                            self.pre_credit[row[0]] = row[1]
                else:
                    for i in xrange(8):
                        attrs.append(row[i])
                counter += 1
        print 'The length of feature:', len(feature), feature

        featureL = len(feature)
        for i in xrange(8):
            if types[i] == 'str':
                featureL += tags[i]-2
                print i, tags[i] - 1, attrs[i], dicts[i]
        print 'The length of feature:', featureL

        t_len = featureL
        counter = 0
        for key in self.cur_pos:
            if self.cur_pos[key] is not None:
                feature, _ = self.cur_pos[key]
                ans = []
                for i in xrange(3, 8):
                    if types[i] == 'str':
                        ans += oneHot(tags[i], feature[i-3])
                    else:
                        if maxN[i] != 0.0:
                            ans.append(feature[i-3]/maxN[i])
                        else:
                            ans.append(feature[i-3])
                self.cur_pos[key] = ans
            else:
                self.cur_pos[key] = [0.0 for k in xrange(t_len)]
                counter += 1
        print 'The # of None in self.cur_pos:{:.4f}'.format(counter*1.0/len(self.cur_pos))
        print 'Pos feature:', len(ans), t_len, ans

        # ***************** read the credit_card_balance.csv ********************************#
        print '*'*15, 'Loading credit_card_balance', '*'*15
        attrs = []
        types = [None for i in xrange(23)]
        dicts = [dict() for i in xrange(23)]
        tags = [1 for i in xrange(23)]
        maxN = [0.0 for i in xrange(23)]
        with open('data/credit_card_balance.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    feature = []
                    for i in xrange(23):
                        if i > 1:
                            if len(row[i]) == 0:
                                feature.append(0.0)
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
                                tmp = float(row[i])
                                maxN[i] = max(maxN[i], abs(tmp))
                                feature.append(tmp)
                    if row[0] in self.pre_credit:
                        sk_id_curr = self.pre_credit[row[0]]
                        ans = self.cur_pos[sk_id_curr]
                        if row[1] != sk_id_curr:
                            print 'Prev_id is not one-to-one mapping to sk_id_curr'
                        if ans is not None:
                            if len(ans) == 2:
                                continue
                            if len(ans) != t_len:
                                ans += [0.0 for k in xrange(len(ans), t_len)]
                            self.cur_pos[row[1]] = [ans, feature]
                        else:
                            self.cur_pos[row[1]] = [[0.0 for k in xrange(t_len)], feature]
                else:
                    for i in xrange(23):
                        attrs.append(row[i])
                counter += 1
        print 'The length of feature:', len(feature), feature

        featureL = len(feature)
        for i in xrange(23):
            if types[i] == 'str':
                featureL += tags[i]-2
                print i, tags[i] - 1, attrs[i], dicts[i]
        print 'The length of feature:', featureL

        t_len += featureL
        for key in self.cur_pos:
            item = self.cur_pos[key]
            if len(item) == 2:
                ans, feature = item
                for i in xrange(2, 23):
                    if types[i] == 'str':
                        ans += oneHot(tags[i], feature[i-2])
                    else:
                        if maxN[i] != 0.0:
                            ans.append(feature[i-2]/maxN[i])
                        else:
                            ans.append(feature[i-2])
            else:
                ans = item + [0.0 for k in xrange(len(item), t_len)]
            self.cur_pos[key] = ans
        print 'Credit feature:', len(ans), t_len, ans

        # ***************** read the previous_application.csv ********************************#
        print '*'*15, 'Loading previous_application', '*'*15
        attrs = []
        types = [None for i in xrange(37)]
        dicts = [dict() for i in xrange(37)]
        tags = [1 for i in xrange(37)]
        maxN = [0.0 for i in xrange(37)]
        with open('data/previous_application.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    feature = []
                    for i in xrange(37):
                        if i > 1:
                            if len(row[i]) == 0:
                                feature.append(0.0)
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
                                tmp = float(row[i])
                                maxN[i] = max(maxN[i], abs(tmp))
                                feature.append(tmp)
                    if row[0] in self.pre_credit:
                        sk_id_curr = self.pre_credit[row[0]]
                        ans = self.cur_pos[sk_id_curr]
                        if row[1] != sk_id_curr:
                            print 'Prev_id is not one-to-one mapping to sk_id_curr'
                        if ans is not None:
                            if len(ans) == 2:
                                continue
                            if len(ans) != t_len:
                                ans += [0.0 for k in xrange(len(ans), t_len)]
                            self.cur_pos[row[1]] = [ans, feature]
                        else:
                            self.cur_pos[row[1]] = [[0.0 for k in xrange(t_len)], feature]
                else:
                    for i in xrange(37):
                        attrs.append(row[i])
                counter += 1
        print 'The length of feature:', len(feature), feature

        featureL = len(feature)
        for i in xrange(37):
            if types[i] == 'str':
                featureL += tags[i]-2
                print i, tags[i] - 1, attrs[i], dicts[i]
        print 'The length of feature:', featureL

        t_len += featureL
        for key in self.cur_pos:
            item = self.cur_pos[key]
            if len(item) == 2:
                ans, feature = item
                for i in xrange(2, 37):
                    if types[i] == 'str':
                        ans += oneHot(tags[i], feature[i-2])
                    else:
                        if maxN[i] != 0.0:
                            ans.append(feature[i-2]/maxN[i])
                        else:
                            ans.append(feature[i-2])
            else:
                ans = item + [0.0 for k in xrange(len(item), t_len)]
            self.cur_pos[key] = ans
        print 'Previous feature:', len(ans), t_len, ans

        # ***************** read the installments_payments.csv ********************************#
        print '*'*15, 'Loading installments_payments', '*'*15
        attrs = []
        types = [None for i in xrange(8)]
        dicts = [dict() for i in xrange(8)]
        tags = [1 for i in xrange(8)]
        maxN = [0.0 for i in xrange(8)]
        with open('data/installments_payments.csv', 'r') as f:
            reader = csv.reader(f)
            counter = 0
            for row in reader:
                if counter:
                    feature = []
                    for i in xrange(8):
                        if i > 1:
                            if len(row[i]) == 0:
                                feature.append(0.0)
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
                                tmp = float(row[i])
                                maxN[i] = max(maxN[i], abs(tmp))
                                feature.append(tmp)
                    if row[0] in self.pre_credit:
                        sk_id_curr = self.pre_credit[row[0]]
                        ans = self.cur_pos[sk_id_curr]
                        if row[1] != sk_id_curr:
                            print 'Prev_id is not one-to-one mapping to sk_id_curr'
                        if ans is not None:
                            if len(ans) == 2:
                                continue
                            if len(ans) != t_len:
                                ans += [0.0 for k in xrange(len(ans), t_len)]
                            self.cur_pos[row[1]] = [ans, feature]
                        else:
                            self.cur_pos[row[1]] = [[0.0 for k in xrange(t_len)], feature]
                else:
                    for i in xrange(8):
                        attrs.append(row[i])
                counter += 1
        print 'The length of feature:', len(feature), feature

        featureL = len(feature)
        for i in xrange(8):
            if types[i] == 'str':
                featureL += tags[i]-2
                print i, tags[i] - 1, attrs[i], dicts[i]
        print 'The length of feature:', featureL

        t_len += featureL
        for key in self.cur_pos:
            item = self.cur_pos[key]
            if len(item) == 2:
                ans, feature = item
                for i in xrange(2, 8):
                    if types[i] == 'str':
                        ans += oneHot(tags[i], feature[i-2])
                    else:
                        if maxN[i] != 0.0:
                            ans.append(feature[i-2]/maxN[i])
                        else:
                            ans.append(feature[i-2])
            else:
                ans = item + [0.0 for k in xrange(len(item), t_len)]
            self.cur_pos[key] = ans
        print 'installments feature:', len(ans), t_len, ans
        print 'Total lenght of the rest of features:', t_len

        lens = []
        for key in self.cur_pos:
            if self.cur_pos[key] is not None:
                lens.append(len(self.cur_pos[key]))
        print np.max(lens), np.min(lens), np.mean(lens)

    def getTVT(self):
        self.loadApplication()
        print len(self.trn), len(self.val), len(self.tst)
        return [dataVal(self.trn), dataVal(self.val), dataVal(self.tst)]

# d = readCSV(0, 0)
# d.loadPos()