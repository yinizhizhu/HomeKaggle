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

        pdb.set_trace()

        ########### Middle layers

        self.middle_layers = dict()

        for key_1 in feature_grouping.keys():
            for key_2 in feature_grouping.keys():

                bi_layer = nn.Bilinear(self.base_out_sizes[key_1], 
                    self.base_out_sizes[key_2], model_params[3], bias=True)

                layer_key = key_1 + '-' + key_2

                self.middle_layers[layer_key] = bi_layer

                self.add_module(name='middle_layer_{}'.format(
                    layer_key), module=bi_layer)


        ########### Upper layers

        upper_in_size = model_params[3] * len(self.middle_layers.keys())
        upper_in_size += len(self.critical_feats)

        self.upper_layers = nn.Sequential(
                nn.Linear(upper_in_size, model_params[4]),
                nn.PReLU(),
                nn.Linear(model_params[4], 2),
            )



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

        base_layer_outs = dict()
        for key, value in embed_features.items():
                base_layer_outs[key] = self.base_layers[key](value)

        ########################

        middle_layer_out = []

        for key_1, value_1 in base_layer_outs.items():
            for key_2, value_2 in base_layer_outs.items():

                layer_key = key_1 + '-' + key_2

                middle_layer_out.append(
                    self.middle_layers[layer_key](value_1, value_2))

        ########################

        # middle_layer_out = middle_layer_out

        for key in self.critical_feats:
            middle_layer_out.append(features[key])

        middle_layer_out = torch.cat(middle_layer_out, dim=1)

        upper_layer_out = self.upper_layers(middle_layer_out)

        ########################


        return upper_layer_out



# {'app': CreditNet(
#   (embed_layer_car_features_0): Linear(in_features=2, out_features=2, bias=False)
#   (embed_layer_car_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_app_features_0): Linear(in_features=2, out_features=2, bias=False)
#   (embed_layer_app_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_app_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_app_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_0): Linear(in_features=3, out_features=3, bias=False)
#   (embed_layer_doc_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_4): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_5): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_6): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_7): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_8): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_9): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_10): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_11): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_12): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_13): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_14): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_15): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_16): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_doc_features_17): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_social_features_0): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_social_features_1): Linear(in_features=7, out_features=3, bias=False)
#   (embed_layer_social_features_2): Linear(in_features=5, out_features=3, bias=False)
#   (embed_layer_social_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_0): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_4): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_5): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_6): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_7): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_8): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_9): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_10): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_11): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_12): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_13): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_14): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_15): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_16): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_17): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_18): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_19): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_20): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_21): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_22): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_23): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_24): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_25): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_26): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_27): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_28): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_29): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_30): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_31): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_32): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_33): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_34): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_35): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_36): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_37): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_38): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_39): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_40): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_41): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_42): Linear(in_features=4, out_features=3, bias=False)
#   (embed_layer_housing_detail_features_43): Linear(in_features=3, out_features=3, bias=False)
#   (embed_layer_housing_detail_features_44): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_detail_features_45): Linear(in_features=7, out_features=3, bias=False)
#   (embed_layer_housing_detail_features_46): Linear(in_features=2, out_features=2, bias=False)
#   (embed_layer_enq_freq_features_0): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_enq_freq_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_enq_freq_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_enq_freq_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_enq_freq_features_4): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_enq_freq_features_5): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_person_features_0): Linear(in_features=2, out_features=2, bias=False)
#   (embed_layer_person_features_1): Linear(in_features=5, out_features=3, bias=False)
#   (embed_layer_person_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_person_features_3): Linear(in_features=2, out_features=2, bias=False)
#   (embed_layer_person_features_4): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_person_features_5): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_social_default_features_0): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_social_default_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_social_default_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_social_default_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_process_features_0): Linear(in_features=7, out_features=3, bias=False)
#   (embed_layer_process_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_features_0): Linear(in_features=2, out_features=2, bias=False)
#   (embed_layer_housing_features_1): Linear(in_features=6, out_features=3, bias=False)
#   (embed_layer_housing_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_housing_features_3): Linear(in_features=2, out_features=2, bias=False)
#   (embed_layer_contact_features_0): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_contact_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_contact_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_contact_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_contact_features_4): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_contact_features_5): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_contact_features_6): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_matching_features_0): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_matching_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_matching_features_2): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_matching_features_3): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_matching_features_4): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_matching_features_5): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_occupation_features_0): Linear(in_features=7, out_features=3, bias=False)
#   (embed_layer_occupation_features_1): Linear(in_features=1, out_features=1, bias=False)
#   (embed_layer_occupation_features_2): Linear(in_features=18, out_features=3, bias=False)
#   (embed_layer_occupation_features_3): Linear(in_features=58, out_features=3, bias=False)
#   (base_layer_car_features): Sequential(
#     (0): Linear(in_features=3, out_features=3, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_app_features): Sequential(
#     (0): Linear(in_features=5, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_doc_features): Sequential(
#     (0): Linear(in_features=20, out_features=10, bias=True)
#     (1): PReLU(num_parameters=1)
#     (2): Linear(in_features=10, out_features=8, bias=True)
#     (3): PReLU(num_parameters=1)
#   )
#   (base_layer_social_features): Sequential(
#     (0): Linear(in_features=8, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_housing_detail_features): Sequential(
#     (0): Linear(in_features=54, out_features=27, bias=True)
#     (1): PReLU(num_parameters=1)
#     (2): Linear(in_features=27, out_features=13, bias=True)
#     (3): PReLU(num_parameters=1)
#     (4): Linear(in_features=13, out_features=8, bias=True)
#     (5): PReLU(num_parameters=1)
#   )
#   (base_layer_enq_freq_features): Sequential(
#     (0): Linear(in_features=6, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_person_features): Sequential(
#     (0): Linear(in_features=10, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_social_default_features): Sequential(
#     (0): Linear(in_features=4, out_features=4, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_process_features): Sequential(
#     (0): Linear(in_features=4, out_features=4, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_housing_features): Sequential(
#     (0): Linear(in_features=8, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_contact_features): Sequential(
#     (0): Linear(in_features=7, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_matching_features): Sequential(
#     (0): Linear(in_features=6, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (base_layer_occupation_features): Sequential(
#     (0): Linear(in_features=10, out_features=8, bias=True)
#     (1): PReLU(num_parameters=1)
#   )
#   (middle_layer_car_features-car_features): Bilinear(in1_features=3, in2_features=3, out_features=2, bias=True)
#   (middle_layer_car_features-app_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-doc_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-social_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-housing_detail_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-enq_freq_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-person_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-social_default_features): Bilinear(in1_features=3, in2_features=4, out_features=2, bias=True)
#   (middle_layer_car_features-process_features): Bilinear(in1_features=3, in2_features=4, out_features=2, bias=True)
#   (middle_layer_car_features-housing_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-contact_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-matching_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_car_features-occupation_features): Bilinear(in1_features=3, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_app_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_app_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_app_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_app_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_doc_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_doc_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_doc_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_doc_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_social_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_social_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_social_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_detail_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_enq_freq_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_person_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_person_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_person_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_person_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-car_features): Bilinear(in1_features=4, in2_features=3, out_features=2, bias=True)
#   (middle_layer_social_default_features-app_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-doc_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-social_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-housing_detail_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-enq_freq_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-person_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-social_default_features): Bilinear(in1_features=4, in2_features=4, out_features=2, bias=True)
#   (middle_layer_social_default_features-process_features): Bilinear(in1_features=4, in2_features=4, out_features=2, bias=True)
#   (middle_layer_social_default_features-housing_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-contact_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-matching_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_social_default_features-occupation_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-car_features): Bilinear(in1_features=4, in2_features=3, out_features=2, bias=True)
#   (middle_layer_process_features-app_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-doc_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-social_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-housing_detail_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-enq_freq_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-person_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-social_default_features): Bilinear(in1_features=4, in2_features=4, out_features=2, bias=True)
#   (middle_layer_process_features-process_features): Bilinear(in1_features=4, in2_features=4, out_features=2, bias=True)
#   (middle_layer_process_features-housing_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-contact_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-matching_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_process_features-occupation_features): Bilinear(in1_features=4, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_housing_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_housing_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_housing_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_housing_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_contact_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_contact_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_contact_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_contact_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_matching_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_matching_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_matching_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_matching_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-car_features): Bilinear(in1_features=8, in2_features=3, out_features=2, bias=True)
#   (middle_layer_occupation_features-app_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-doc_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-social_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-housing_detail_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-enq_freq_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-person_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-social_default_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_occupation_features-process_features): Bilinear(in1_features=8, in2_features=4, out_features=2, bias=True)
#   (middle_layer_occupation_features-housing_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-contact_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-matching_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (middle_layer_occupation_features-occupation_features): Bilinear(in1_features=8, in2_features=8, out_features=2, bias=True)
#   (upper_layers): Sequential(
#     (0): Linear(in_features=341, out_features=64, bias=True)
#     (1): PReLU(num_parameters=1)
#     (2): Linear(in_features=64, out_features=2, bias=True)
#   )
# )}
