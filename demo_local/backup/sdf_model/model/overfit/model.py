#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

import os 
from pathlib import Path
import time 

from sdf_model.model import base_pl

from sdf_model.model.archs.decoders.deepsdf_arch import DeepSdfArch

from sdf_model.utils import mesh, evaluate

class Overfit(base_pl.Model):
    def __init__(self, specs):
        super().__init__(specs)
        
        #encoder_specs = self.specs["EncoderSpecs"]
        self.latent_size = 0 #encoder_specs["latent_size"]
        
        decoder_specs = self.specs["DecoderSpecs"]
        self.decoder_hidden_dim = decoder_specs["hidden_dim"]
        self.skip_connection = decoder_specs["skip_connection"]
        self.geo_init = decoder_specs["geo_init"]
        self.weight_norm = decoder_specs["weight_norm"]
        self.tanh_act = decoder_specs["tanh_act"]
        self.dropout = decoder_specs["dropout_prob"]

        lr_specs = self.specs["LearningRate"]
        #self.lr_enc_init = lr_specs["enc_init"]
        self.lr_dec_init = lr_specs["dec_init"]
        self.lr_step = lr_specs["step_size"]
        self.lr_gamma = lr_specs["gamma"]

        self.samples_per_mesh = self.specs["SampPerMesh"]
        self.beta = self.specs.get("Beta", 0)
        #print("surface weight: ",self.beta)

        self.build_model()

        self.save_hyperparameters()


    def build_model(self):
        
        #ad = AutoDecoder(self.num_objects, self.latent_size)
        #self.encoder = ad.build_model()
        
        self.decoder = DeepSdfArch(self.latent_size, self.decoder_hidden_dim, geo_init=self.geo_init, 
                                  skip_connection=self.skip_connection, weight_norm=self.weight_norm,
                                  tanh_act=self.tanh_act)


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(
            [
                # {
                #     "params": self.encoder.parameters(),
                #     "lr": self.lr_enc_init # 1e-3
                # },
                {
                    "params": self.decoder.parameters(),
                    "lr": self.lr_dec_init # 1e-5*batch size
                }
            ]
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, self.lr_step, self.lr_gamma)

        return [optimizer], [lr_scheduler]

 
    # context and queries from labeled, unlabeled data, respectively 
    def training_step(self, x, batch_idx):

        indices = x['indices']
        xyz = x['sdf_xyz'].float()
        gt_sdf = x['gt_sdf'].float()

        #print("xyz, gt shape: ",xyz.shape, gt_sdf.shape)

        xyz = xyz.reshape(-1,3)
        gt_sdf = gt_sdf.reshape(-1)

        pred_sdf = self.decoder(xyz)
        #pred_sdf = torch.clamp(pred_sdf, -0.1, 0.1)
        #gt_sdf = torch.clamp(gt_sdf, -0.1, 0.1)
        
        l1_loss = F.l1_loss(pred_sdf, gt_sdf, reduction='none')

        surface_weight = torch.exp(-self.beta * torch.abs(gt_sdf))
       
        
        return torch.mean(l1_loss * surface_weight)
        
    def reconstruct(self, model, gt_pc, resolution, eval_dir):
        recon_samplesize_param = resolution
        recon_batch = int(2 ** 19)

        
        model.eval() 

        with torch.no_grad():
            
            mesh.create_mesh_single(model, eval_dir, recon_samplesize_param, recon_batch)
            
            # try:
            cd = evaluate.out(gt_pc, "sdf_model/output/reconstruct.ply") if gt_pc is not None else None
            # except Exception as e:
            #     print(e)
        return cd

        

    






        