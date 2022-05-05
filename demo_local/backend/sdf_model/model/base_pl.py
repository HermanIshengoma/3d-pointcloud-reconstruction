#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import json
import os 
from pathlib import Path
import time 

import pandas as pd 
import numpy as np 

from sdf_model.utils import test_opt, evaluate, mesh


class Model(pl.LightningModule):
    # specs is a json filepath that contains the specifications for the experiment 
    def __init__(self, specs):
        super().__init__()

        if type(specs) == dict:
            self.specs = specs

        else:
            if not os.path.isfile(specs):
                raise Exception("The specifications at {} do not exist!!".format(specs))
            self.specs = json.load(open(specs))

    # forward is used for inference/predictions
    # def forward(self, x):
    #     raise NotImplementedError
    # training_step replaces the original forward function 
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def configure_optimizer(self):
        raise NotImplementedError

 
    def recon(self, model, pc, mesh_name, eval_dir):
        sampled_pc = pc[:,torch.randperm(pc.shape[1])][:,0:1024]
        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")
            
            recon_samplesize_param = 256
            recon_batch = int(2 ** 18)

            try:
                mesh.create_mesh_ad(model, sampled_pc, mesh_filename, recon_samplesize_param, recon_batch)
            except Exception as e:
                print(e)
            try:
                evaluate.main(pc, mesh_filename, evaluate_filename, mesh_name)
            except Exception as e:
                print(e)

    def reconstruct(self, model, test_data, eval_dir, test_opt=False, do_recon=True, do_evaluate=True):

        if not do_recon and not do_evaluate:
            print("at least one of args.reconstruct and args.evaluate should be true!")
            return

        gt_pc = test_data['point_cloud'].float()
        sampled_pc = gt_pc[:,torch.randperm(gt_pc.shape[1])][:,0:1024]
        #print("pc shapes: ",gt_pc.shape, sampled_pc.shape)

        if test_opt:
            t1 = time.time()
            model = fast_opt(model, sampled_pc)
            print("time for test opt: ", time.time()-t1)

        l, latent = self.deepsdf_opt(model, 800, 256, test_data["xyz"], test_data["gt_sdf"])

        print("loss: ",l)
        model.eval() 

        if "EvaluationSpecs" in self.specs:
            recon_samplesize_param = self.specs["EvaluationSpecs"]["recon_samplesize_param"]
            recon_batch = self.specs["EvaluationSpecs"]["recon_batch"]
        else:
            # defaults in deepsdf repo
            recon_samplesize_param = 256
            recon_batch = int(2 ** 18)

        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")
            
            mesh_name = test_data["mesh_name"]

            if do_recon:
                #try:           
                mesh.create_mesh_idx(model, test_data["indices"], mesh_filename, recon_samplesize_param, recon_batch)
                #mesh.create_mesh_ad(model, latent, mesh_filename, recon_samplesize_param, recon_batch)
                #mesh.create_mesh_clean(model, sampled_pc, mesh_filename, recon_samplesize_param, recon_batch)
                #except Exception as e:
                #    print(e)
            if do_evaluate:
                try:
                    evaluate.main(gt_pc, mesh_filename, evaluate_filename, mesh_name)
                except Exception as e:
                    print(e)


    def deepsdf_opt(
        self,
        decoder,
        num_iterations,
        latent_size,
        sdf_xyz,
        gt_sdf,
        num_samples=30000,
        lr=5e-4,
    ):
        
        latent = torch.ones(1, latent_size).normal_(mean=0, std=0.1).cuda()
        latent.requires_grad = True

        optimizer = torch.optim.Adam([latent], lr=lr)

        loss_l1 = torch.nn.L1Loss()

        for e in range(num_iterations):

            decoder.eval()
            
            xyz = sdf_xyz.cuda().squeeze()
            sdf_gt = gt_sdf.cuda().squeeze()

            sdf_gt = torch.clamp(sdf_gt, -0.1,0.1)

            optimizer.zero_grad()

            latent_inputs = latent.expand(xyz.shape[0], -1)

            inputs = torch.cat([latent_inputs, xyz], -1).cuda().float()

            pred_sdf = decoder.decoder(inputs)

            # TODO: why is this needed?
            if e == 0:
                pred_sdf = decoder.decoder(inputs)

            pred_sdf = torch.clamp(pred_sdf, -0.1,0.1).squeeze()

            loss = loss_l1(pred_sdf, sdf_gt)

            loss.backward()
            optimizer.step()

            loss_num = loss.cpu().data.numpy()

        return loss_num, latent



        
        