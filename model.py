import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb


# 
class CreditNet(nn.Module):
    def __init__(self, feature_grouping, critical_feats, model_params):
        super(CreditNet, self).__init__()

        self.feature_grouping = feature_grouping
        self.critical_feats = critical_feats
        self.model_params = model_params

        for feature in self.critical_feats:
            self.feature_grouping.pop(feature)

        ########### Embed layers 
        self.embed_layers = dict()
        self.embed_out_sizes = {k : 0 for k, v in feature_grouping.items()}

        for key, value in feature_grouping.items():
            modules = []
            for index, item in enumerate(value):
                embed_size = min(len(item), model_params[0])

                embedding = nn.Linear(len(item), embed_size, bias=False)

                self.add_module(name='embed_layer_{}_{}'.format(key, index), 
                                module=embedding)

                modules.append(embedding)
                self.embed_out_sizes[key] += embed_size

            self.embed_layers[key] = modules

        ########### Base layers

        self.base_layers = dict()
        self.base_out_sizes = {k : 0 for k, v in feature_grouping.items()}

        for key in feature_grouping.keys():

            embed_size = self.embed_out_sizes[key]

            module_list = []

            if embed_size <= (model_params[1] // 2):    # 3 -> 3
                module_list.append(nn.Linear(embed_size, embed_size))
                module_list.append(nn.PReLU())

                self.base_out_sizes[key] = embed_size

            else:
                while embed_size > model_params[1] * model_params[2]:   #10 -> 8    20 -> 10 -> 8    54 -> 27 -> 13 -> 8
                    module_list.append(nn.Linear(embed_size, 
                        embed_size // model_params[2]))

                    module_list.append(nn.PReLU())
                    embed_size = embed_size // model_params[2]

                module_list.append(nn.Linear(embed_size, model_params[1]))
                module_list.append(nn.PReLU())

                self.base_out_sizes[key] = model_params[1]

            module = nn.Sequential(*module_list)

            self.add_module(name='base_layer_{}'.format(key), module=module)

            self.base_layers[key] = module

        self.base_out_size_sum = sum([v for v in self.base_out_sizes.values()])

        ########### Middle layers
        self.middle_layers = nn.Sequential(
                nn.Linear(self.base_out_size_sum, model_params[3]),
                nn.PReLU(),
                nn.Linear(model_params[3], model_params[4]),
                nn.PReLU()
            )

        ########### Upper layers
        self.upper_layers = nn.Sequential(
                nn.Linear(model_params[4] + len(self.critical_feats), 
                    model_params[5]),
                nn.PReLU(),
                nn.Linear(model_params[5], 2),
            )

        #self.head = nn.LogSoftmax(dim=1)


    def forward(self, features):


        ########################
        embed_features = {k : [] for k in features.keys()
                            if k not in self.critical_feats}

        for key, value in self.feature_grouping.items():
            for index, item in enumerate(value):

                one_features = features[key][:,item]
                one_outs = self.embed_layers[key][index](one_features)

                embed_features[key].append(one_outs)
        
            embed_features[key] = torch.cat(embed_features[key], dim=1)

        ########################
        base_layer_outs = []
        for key, value in embed_features.items():
                base_layer_outs.append(self.base_layers[key](value))

        base_layer_outs = torch.cat(base_layer_outs, dim=1)

        ########################

        middle_layer_out = self.middle_layers(base_layer_outs)

        ########################

        upper_layer_in = [middle_layer_out]

        for key in self.critical_feats:
            upper_layer_in.append(features[key])

        upper_layer_in = torch.cat(upper_layer_in, dim=1)

        upper_layer_out = self.upper_layers(upper_layer_in)

        ########################

        #final_out = self.head(upper_layer_out)

        return upper_layer_out


# CreditNet(
#   (embed_layer_app_features_0): Linear(in_features=2, out_features=2, bias=True)
#   (embed_layer_app_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_app_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_app_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_person_features_0): Linear(in_features=2, out_features=2, bias=True)
#   (embed_layer_person_features_1): Linear(in_features=5, out_features=3, bias=True)
#   (embed_layer_person_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_person_features_3): Linear(in_features=2, out_features=2, bias=True)
#   (embed_layer_person_features_4): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_person_features_5): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_occupation_features_0): Linear(in_features=7, out_features=3, bias=True)
#   (embed_layer_occupation_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_occupation_features_2): Linear(in_features=18, out_features=3, bias=True)
#   (embed_layer_occupation_features_3): Linear(in_features=58, out_features=3, bias=True)
#   (embed_layer_car_features_0): Linear(in_features=2, out_features=2, bias=True)
#   (embed_layer_car_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_features_0): Linear(in_features=2, out_features=2, bias=True)
#   (embed_layer_housing_features_1): Linear(in_features=6, out_features=3, bias=True)
#   (embed_layer_housing_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_features_3): Linear(in_features=2, out_features=2, bias=True)
#   (embed_layer_contact_features_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_contact_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_contact_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_contact_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_contact_features_4): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_contact_features_5): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_contact_features_6): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_social_features_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_social_features_1): Linear(in_features=7, out_features=3, bias=True)
#   (embed_layer_social_features_2): Linear(in_features=5, out_features=3, bias=True)
#   (embed_layer_social_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_social_default_features_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_social_default_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_social_default_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_social_default_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_matching_features_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_matching_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_matching_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_matching_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_matching_features_4): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_matching_features_5): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_0): Linear(in_features=3, out_features=3, bias=True)
#   (embed_layer_doc_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_4): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_5): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_6): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_7): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_8): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_9): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_10): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_11): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_12): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_13): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_14): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_15): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_16): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_doc_features_17): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_enq_freq_features_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_enq_freq_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_enq_freq_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_enq_freq_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_enq_freq_features_4): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_enq_freq_features_5): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_process_features_0): Linear(in_features=7, out_features=3, bias=True)
#   (embed_layer_process_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_1): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_2): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_3): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_4): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_5): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_6): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_7): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_8): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_9): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_10): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_11): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_12): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_13): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_14): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_15): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_16): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_17): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_18): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_19): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_20): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_21): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_22): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_23): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_24): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_25): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_26): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_27): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_28): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_29): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_30): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_31): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_32): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_33): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_34): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_35): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_36): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_37): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_38): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_39): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_40): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_41): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_42): Linear(in_features=4, out_features=3, bias=True)
#   (embed_layer_housing_detail_features_43): Linear(in_features=3, out_features=3, bias=True)
#   (embed_layer_housing_detail_features_44): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_housing_detail_features_45): Linear(in_features=7, out_features=3, bias=True)
#   (embed_layer_housing_detail_features_46): Linear(in_features=2, out_features=2, bias=True)
#   (embed_layer_EXT_SOURCE_1_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_EXT_SOURCE_2_0): Linear(in_features=1, out_features=1, bias=True)
#   (embed_layer_EXT_SOURCE_3_0): Linear(in_features=1, out_features=1, bias=True)
#   (base_layer_app_features): Sequential(
#     (0): Linear(in_features=5, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_person_features): Sequential(
#     (0): Linear(in_features=10, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_occupation_features): Sequential(
#     (0): Linear(in_features=10, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_car_features): Sequential(
#     (0): Linear(in_features=3, out_features=3, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_housing_features): Sequential(
#     (0): Linear(in_features=8, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_contact_features): Sequential(
#     (0): Linear(in_features=7, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_social_features): Sequential(
#     (0): Linear(in_features=8, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_social_default_features): Sequential(
#     (0): Linear(in_features=4, out_features=4, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_matching_features): Sequential(
#     (0): Linear(in_features=6, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_doc_features): Sequential(
#     (0): Linear(in_features=20, out_features=6, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=6, out_features=5, bias=True)
#     (3): ReLU()
#   )
#   (base_layer_enq_freq_features): Sequential(
#     (0): Linear(in_features=6, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_process_features): Sequential(
#     (0): Linear(in_features=4, out_features=4, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_housing_detail_features): Sequential(
#     (0): Linear(in_features=54, out_features=18, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=18, out_features=6, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=6, out_features=5, bias=True)
#     (5): ReLU()
#   )
#   (base_layer_EXT_SOURCE_1): Sequential(
#     (0): Linear(in_features=1, out_features=1, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_EXT_SOURCE_2): Sequential(
#     (0): Linear(in_features=1, out_features=1, bias=True)
#     (1): ReLU()
#   )
#   (base_layer_EXT_SOURCE_3): Sequential(
#     (0): Linear(in_features=1, out_features=1, bias=True)
#     (1): ReLU()
#   )
#   (upper_layers): Sequential(
#     (0): Linear(in_features=64, out_features=256, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=256, out_features=128, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=128, out_features=64, bias=True)
#     (5): ReLU()
#     (6): Linear(in_features=64, out_features=2, bias=True)
#     (7): Softmax()
#   )
# )


