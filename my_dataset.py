import os
import torch
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import Dataset

import pdb

data_dir='../input'

_app_grouping_map = {

    'id': [
        'SK_ID_CURR'
    ],

    'app_features' : [
        'NAME_CONTRACT_TYPE',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE'
    ],

    'person_features' : [
        'CODE_GENDER',
        'NAME_EDUCATION_TYPE',
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'DAYS_REGISTRATION',
        'DAYS_ID_PUBLISH',
    ],

    'occupation_features' : [
        'NAME_INCOME_TYPE',
        'AMT_INCOME_TOTAL',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE'
    ],

    'car_features' : [
        'FLAG_OWN_CAR',
        'OWN_CAR_AGE',
    ],

    'housing_features' : [
        'FLAG_OWN_REALTY',
        'NAME_HOUSING_TYPE',
        'REGION_POPULATION_RELATIVE',
        'REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY',
    ],

    'contact_features' : [
        'FLAG_MOBIL',
        'FLAG_EMP_PHONE',
        'FLAG_WORK_PHONE',
        'FLAG_CONT_MOBILE',
        'FLAG_PHONE',
        'FLAG_EMAIL',
        'DAYS_LAST_PHONE_CHANGE',
    ],

    'social_features' : [
        'CNT_CHILDREN',
        'NAME_TYPE_SUITE',
        'NAME_FAMILY_STATUS',
        'CNT_FAM_MEMBERS',
    ],

    'social_default_features' : [
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
    ],

    'matching_features' : [
        'REG_REGION_NOT_LIVE_REGION',
        'REG_REGION_NOT_WORK_REGION',
        'LIVE_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY',
        'LIVE_CITY_NOT_WORK_CITY',
    ],

    'doc_features' : [
        'FLAG_DOCUMENT_2',
        'FLAG_DOCUMENT_3',
        'FLAG_DOCUMENT_4',
        'FLAG_DOCUMENT_5',
        'FLAG_DOCUMENT_6',
        'FLAG_DOCUMENT_7',
        'FLAG_DOCUMENT_8',
        'FLAG_DOCUMENT_9',
        'FLAG_DOCUMENT_10',
        'FLAG_DOCUMENT_11',
        'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_13',
        'FLAG_DOCUMENT_14',
        'FLAG_DOCUMENT_15',
        'FLAG_DOCUMENT_16',
        'FLAG_DOCUMENT_17',
        'FLAG_DOCUMENT_18',
        'FLAG_DOCUMENT_19',
        'FLAG_DOCUMENT_20',
        'FLAG_DOCUMENT_21',
    ],

    'enq_freq_features' : [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
    ],

    'process_features' : [
        'WEEKDAY_APPR_PROCESS_START',
        'HOUR_APPR_PROCESS_START',
    ],

    'housing_detail_features' : [
        'APARTMENTS_AVG',
        'BASEMENTAREA_AVG',
        'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG',
        'COMMONAREA_AVG',
        'ELEVATORS_AVG',
        'ENTRANCES_AVG',
        'FLOORSMAX_AVG',
        'FLOORSMIN_AVG',
        'LANDAREA_AVG',
        'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG',
        'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG',
        'APARTMENTS_MODE',
        'BASEMENTAREA_MODE',
        'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE',
        'COMMONAREA_MODE',
        'ELEVATORS_MODE',
        'ENTRANCES_MODE',
        'FLOORSMAX_MODE',
        'FLOORSMIN_MODE',
        'LANDAREA_MODE',
        'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE',
        'APARTMENTS_MEDI',
        'BASEMENTAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI',
        'COMMONAREA_MEDI',
        'ELEVATORS_MEDI',
        'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI',
        'FLOORSMIN_MEDI',
        'LANDAREA_MEDI',
        'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI',
        'NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAREA_MEDI',
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'TOTALAREA_MODE',
        'WALLSMATERIAL_MODE',
        'EMERGENCYSTATE_MODE',
    ],

    'EXT_SOURCE_1' : [
        'EXT_SOURCE_1'
    ],

    'EXT_SOURCE_2' : [
        'EXT_SOURCE_2'
    ],

    'EXT_SOURCE_3' : [
        'EXT_SOURCE_3'
    ],
}

def _search_feature_group(grouping_map, feat):
    result = None
    for key, value in grouping_map.items():
        if feat in value:
            result = key
            break
    return result

def _get_all_raw_feature(grouping_map):
    all_feature = list(grouping_map.values())
    all_feature = [feat for group in all_feature for feat in group]
    return all_feature


# Maybe remove low variance features
# bureau bal un used
# self.bureau_bal_file = os.path.join(data_dir, 'bureau_balance.csv')
# self.bureau_bal = pd.read_csv(self.bureau_bal_file)  # (27299925, 3)

class LoanDataset(object):

    def __init__(self):

        ################# Load Data ################################
        data_dir='../input'

        self.table_names = [
            'application_train',
            'application_test',
            'bureau',
            # 'POS_CASH_balance',
            # 'credit_card_balance',
            # 'previous_application',
            # 'installments_payments'
        ]

        self.all_tables = dict()

        for name in self.table_names:

            file = os.path.join(data_dir, name + '.csv')
            table = pd.read_csv(file)
            self.all_tables[name] = table

            print('{} shape: {}'.format(name, table.shape))


        self.labels = np.array(self.all_tables['application_train']['TARGET'])
        print('Raw training label shape: ', self.labels.shape)

        ################# Grouping dict init ################################

        self.all_grouping_maps = dict()

        for name in self.table_names:

            if name in ['application_train', 'application_test']:
                self.all_grouping_maps[name] = _app_grouping_map
            else:
                table = self.all_tables[name]
                self.all_grouping_maps[name] = {c : [c] for c in table.columns}

        # IF PROVIDED ??

        ################# Feature grouping ################################

        for name in self.table_names:

            grouping_map = self.all_grouping_maps[name]

            self.all_tables[name] = {k : self.all_tables[name][v] 
                for k, v in grouping_map.items()}

        ################# One-hot encoding ################################

        for name in self.table_names:

            self.all_tables[name] = {k : pd.get_dummies(v, dummy_na=False)     # Nan: all zero
                for k, v in self.all_tables[name].items()}

        ############ Align the App training and testing data ##############

        for group in _app_grouping_map.keys(): # keep only columns present in both dataframes
            
            temp_train = self.all_tables['application_train'][group]
            temp_test = self.all_tables['application_test'] [group]

            temp_train, temp_test = temp_train.align(
                temp_test, join = 'inner', axis = 1)

            self.all_tables['application_train'][group] = temp_train
            self.all_tables['application_test'][group] = temp_test

        
        ################# Remove anomaly ################################

        self._remove_anomaly(table_name='application_train',
            anom_feat='DAYS_EMPLOYED', anom_value=365243)

        self._remove_anomaly(table_name='application_test',
            anom_feat='DAYS_EMPLOYED', anom_value=365243)

        ################# Update feature group maps ################################

        # print(self.all_grouping_maps['application_train']['app_features'])

        for name in self.table_names:

            old_grouping_map = dict(self.all_grouping_maps[name])
            all_raw_feat = _get_all_raw_feature(old_grouping_map)

            self.all_grouping_maps[name] = {k : list(v.columns) 
                for k, v in self.all_tables[name].items()}

            # if name == 'application_train':
            #     print(self.all_grouping_maps['application_train']['app_features'])

            index_finder_copy = dict(self.all_grouping_maps[name])  # For finding idx

            # Sub-group one-hot features and to idx
            for raw_feat in all_raw_feat:

                raw_feat_group = _search_feature_group(
                    old_grouping_map, raw_feat)

                feat_list = self.all_grouping_maps[name][raw_feat_group]
                feat_list_temp = list(feat_list)
                feat_sub_list = []

                for idx, feat in enumerate(feat_list):
                    if type(feat) == list:
                        continue
                    if feat.startswith(raw_feat):
                        real_idx = index_finder_copy[raw_feat_group].index(feat)
                        feat_sub_list.append(real_idx)
                        feat_list_temp.remove(feat)

                if feat_sub_list:
                    feat_list_temp.append(feat_sub_list)

                self.all_grouping_maps[name][raw_feat_group] = feat_list_temp

        # print(self.all_grouping_maps['application_train']['app_features'])

        ################ Fill missing data ################
        for group in self.all_grouping_maps['application_train'].keys():

            imputer = Imputer(strategy = 'median')
            imputer.fit(self.all_tables['application_train'][group])

            self.all_tables['application_train'][group] = imputer.transform(
                self.all_tables['application_train'][group])
            self.all_tables['application_test'][group] = imputer.transform(
                self.all_tables['application_test'][group])


        for name in self.table_names:

            if name in ['application_train', 'application_test']:
                continue

            for group in self.all_grouping_maps[name].keys():

                imputer = Imputer(strategy = 'median')
                self.all_tables[name][group] = imputer.fit_transform(
                    self.all_tables[name][group])

        # 242 features
        ############### Scale features ####################
        for group in self.all_grouping_maps['application_train'].keys():

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(self.all_tables['application_train'][group])

            self.all_tables['application_train'][group] = scaler.transform(
                self.all_tables['application_train'][group])
            self.all_tables['application_test'][group] = scaler.transform(
                self.all_tables['application_test'][group])


        for name in self.table_names:

            if name in ['application_train', 'application_test']:
                continue

            for group in self.all_grouping_maps[name].keys():

                scaler = MinMaxScaler(feature_range=(0, 1))
                self.all_tables[name][group] = scaler.fit_transform(
                    self.all_tables[name][group])

        ############## Train Val Split ######################

        listed_app_train = list(self.all_tables['application_train'].items())
        listed_keys = [i[0] for i in listed_app_train]
        listed_values = [i[1] for i in listed_app_train]

        listed_values.append(self.labels)

        split_out = train_test_split(*listed_values, 
                                     test_size=0.1, 
                                     random_state=42)

        self.all_tables['application_train'] = dict()
        self.all_tables['application_val'] = dict()

        for idx, key in enumerate(listed_keys):
            self.all_tables['application_train'][key] = split_out[2 * idx]
            self.all_tables['application_val'][key] = split_out[2 * idx + 1]

        self.train_labels = split_out[-2]
        self.val_labels = split_out[-1]

        def query(self, id):
            pass # TO DO





    def _print_column_num(self):
        print(sum([len(i.columns) for i in self.app_train.values()]))
        print(sum([len(i.columns) for i in self.app_test.values()]))

    def _print_df_shape(self):

        print('\nTraining data shape:\n')
        for key, value in self.app_train.items():
            print(key)
            print(value.shape)

        print('\nTesting data shape:\n')
        for key, value in self.app_test.items():
            print(key)
            print(value.shape)

    def _remove_anomaly(self, table_name, anom_feat, anom_value):

        grouping_map = self.all_grouping_maps[table_name]
        group = _search_feature_group(grouping_map, anom_feat)

        if not group:
            raise Exception('Invalid feature')

        self.all_tables[table_name][group][anom_feat + '_ANOM'] = \
            self.all_tables[table_name][group][anom_feat] == anom_value

        self.all_tables[table_name][group][anom_feat].replace(
                {anom_value: np.nan}, inplace = True)




loan_dataset = LoanDataset()






class LoanDatasetWrapper(Dataset):

    def __init__(self, mode):

        if mode not in ['train', 'test', 'val']:
            raise Exception('Invalid mode')
        else:
            self.mode = mode

        if self.mode == 'train':
            self.data = loan_dataset.all_tables['application_train']
            self.label = loan_dataset.train_labels
        elif self.mode == 'val':
            self.data = loan_dataset.all_tables['application_val']
            self.label = loan_dataset.val_labels
        elif self.mode == 'test':
            self.data = loan_dataset.all_tables['application_test']

    def get_feature_grouping(self):

        grouping_map = dict(loan_dataset.all_grouping_maps['application_train'])
        grouping_map.pop('id')

        return grouping_map

    def __len__(self):
        lens = [i.shape[0] for i in self.data.values()]
        return lens[0]

    def __getitem__(self, idx):

        entry = {k : v[idx] for k, v in self.data.items()}

        if self.mode in ['train', 'val']:
            entry['label'] = self.label[idx]

        return entry


        


# test_dataset = LoanDatasetWrapper(mode='train')
# entry = test_dataset[0]

# pdb.set_trace()








        # Drop useless columns
        # self.app_train = self.app_train.drop(columns = ['SK_ID_CURR'])
        # self.app_test = self.app_test.drop(columns = ['SK_ID_CURR'])
