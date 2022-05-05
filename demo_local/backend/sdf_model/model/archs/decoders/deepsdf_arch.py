#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import sys
import torch.nn.init as init
import numpy as np


class DeepSdfArch(nn.Module):
    def __init__(self, latent_size=256, hidden_dim=512, weight_norm=False, 
                 skip_connection=True, dropout_prob=0.0, tanh_act=False,
                 geo_init=True, input_size=None
                 ):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = latent_size+3 if input_size is None else input_size
        self.skip_connection = skip_connection
        dp = dropout_prob
        self.tanh_act = tanh_act

        skip_dim = hidden_dim+self.input_size if skip_connection else hidden_dim 

        if weight_norm:

            self.block1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(self.input_size, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp)
            )

            self.block2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(skip_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(p=dp)
            )

        else:
            self.block1 = nn.Sequential(
                nn.Linear(self.input_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp)
            )

            self.block2 = nn.Sequential(
                nn.Linear(skip_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dp)
            )


        self.block3 = nn.Linear(hidden_dim, 1)

        if geo_init:
            for m in self.block3.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(hidden_dim), std=0.000001)
                    init.constant_(m.bias, -0.5)

            for m in self.block2.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)

            for m in self.block1.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)




    def forward(self, x):

        block1_out = self.block1(x)

        # skip connection, concat 
        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1) 
        else:
            block2_in = block1_out

        block2_out = self.block2(block2_in)

        out = self.block3(block2_out)

        if self.tanh_act:
            out = nn.Tanh()(out)

        return out.squeeze()

