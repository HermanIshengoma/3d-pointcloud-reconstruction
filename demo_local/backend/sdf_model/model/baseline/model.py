#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import math

import os 
from pathlib import Path
import time 

from sdf_model.model import base_pl
from sdf_model.model.archs.encoders.conv_pointnet import ConvPointnet
from sdf_model.model.archs.decoders.deepsdf_arch import DeepSdfArch

from sdf_model.utils import mesh, evaluate

# final architecture but no meta-learning 
# good for ablation such as pos encoding 
class Baseline(base_pl.Model):
    def __init__(self, specs):
        super().__init__(specs)
        
        encoder_specs = self.specs["EncoderSpecs"]
        self.latent_size = encoder_specs["latent_size"]
        self.latent_hidden_dim = encoder_specs["hidden_dim"]
        self.unet_kwargs = encoder_specs["unet_kwargs"]
        self.plane_resolution = encoder_specs["plane_resolution"]

        decoder_specs = self.specs["DecoderSpecs"]
        self.decoder_hidden_dim = decoder_specs["hidden_dim"]
        self.skip_connection = decoder_specs["skip_connection"]
        self.geo_init = decoder_specs["geo_init"]

        lr_specs = self.specs["LearningRate"]
        self.lr_init = lr_specs["init"]
        self.lr_step = lr_specs["step_size"]
        self.lr_gamma = lr_specs["gamma"]

        # FFN gaussian transformation
        # self.mapping_size = self.specs.get("CoordMapSize", 256)
        # mapping_scale = self.specs.get("CoordScale", 12.0)
        # self.bvals = (torch.normal(0.0,1.0,size=(self.mapping_size,3))* mapping_scale) #np.random.normal(size=[256,3])*12.0
        # self.avals = torch.ones_like(self.bvals[:,0])
        # self.ff_enc = lambda x, a, b: torch.cat([a * torch.sin((2.*math.pi*x) @ b.T), 
        #                                          a * torch.cos((2.*math.pi*x) @ b.T)], axis=-1) / torch.norm(a)


        #np.concatenate([a * np.sin((2.*np.pi*x) @ b.T), a * np.cos((2.*np.pi*x) @ b.T)], axis=-1) / np.linalg.norm(a)

        self.build_model()


    def build_model(self):
        self.encoder = ConvPointnet(c_dim=self.latent_size, hidden_dim=self.latent_hidden_dim, 
                                        plane_resolution=self.plane_resolution,
                                        unet=(self.unet_kwargs is not None), unet_kwargs=self.unet_kwargs)
        
        self.decoder = DeepSdfArch(self.latent_size, self.decoder_hidden_dim, geo_init=self.geo_init, 
                                  skip_connection=self.skip_connection)#, input_size=self.latent_size+self.mapping_size*2)


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), self.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, self.lr_step, self.lr_gamma)

        return [optimizer], [lr_scheduler]

 
    # context and queries from labeled, unlabeled data, respectively 
    def training_step(self, x, batch_idx):

        pc = x['point_cloud']
        xyz = x['sdf_xyz']
        gt = x['gt_sdf']

        shape_vecs = self.encoder(pc, xyz)

        #enc_xyz = self.ff_enc(xyz, self.avals.to(self.device), self.bvals.to(self.device)) # (N,3) -> (N,512)


        decoder_input = torch.cat([shape_vecs, xyz], dim=-1)
        pred_sdf = self.decoder(decoder_input)
        

        # labeled (supervised) loss
        loss = self.labeled_loss(pred_sdf, gt)

        
        return loss
        
        


    def labeled_loss(self, pred_sdf, gt_sdf):

        l1_loss = nn.L1Loss()(pred_sdf.squeeze(), gt_sdf.squeeze())
            
        return l1_loss 

    def reconstruct(self, model, gt_pc, pc_size, resolution, eval_dir):
        recon_samplesize_param = resolution
        recon_batch = int(2 ** 19)

        print("gt pc shape: ",gt_pc.shape)
        sampled_pc = gt_pc[:,torch.randperm(gt_pc.shape[1])[0:pc_size]]
        print("sampled pc shape: ",sampled_pc.shape)
        model.eval() 
        
        
        with torch.no_grad():
            mesh.create_mesh_prior(model, sampled_pc, eval_dir, recon_samplesize_param, recon_batch)

            cd = evaluate.out(gt_pc, "sdf_model/output/reconstruct.ply") if gt_pc is not None else None
            # except Exception as e:
            #     print(e)
        return cd


        

    
