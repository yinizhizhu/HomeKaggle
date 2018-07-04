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
