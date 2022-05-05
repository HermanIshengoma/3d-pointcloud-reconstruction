#!/usr/bin/env python3

import torch
import torch.utils.data 

import os
import json
import time

from sdf_model.model import *



def main(resolution, gt_pc=None):
    
    '''
    This model is only trained on one object and can only reconstruct one object 
    the only variable is the resolution required; no point cloud input required
    the resolution can be any integer but we should set to only 64, 96, 128 in the front end
    '''

    specs_path = "sdf_model/config/model3_specs.json"

    specs = json.load(open(specs_path))

    model = Baseline(specs)

    checkpoint = torch.load("sdf_model/checkpoint/model3.ckpt", map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])

    out_dir = "sdf_model/output/reconstruct"


    print("passed in pc shape: ",gt_pc.shape)
    gt_pc = torch.from_numpy(gt_pc).unsqueeze(0).float()
    cd = model.reconstruct(model, gt_pc, gt_pc.shape[1], resolution, out_dir)

    print("chamfer distance: ",cd)
    return cd


# cd = main(gt_pc, 128)
# print(cd)