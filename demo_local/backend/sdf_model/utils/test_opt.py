#!/usr/bin/env python3

import logging
import json
import numpy as np
import torch.nn as nn
import torch
import os

# test time optimization helper functions 
def fast_preprocess(pc):
    pc = pc.squeeze()
    pc_size = pc.shape[0] # 1024 or 5000
    query_per_point=20

    # grid spanning unit cube to prevent overfitting
    def gen_grid(start, end, num):
        x = np.linspace(start,end,num=num)
        y = np.linspace(start,end,num=num)
        z = np.linspace(start,end,num=num)
        g = np.meshgrid(x,y,z)
        positions = np.vstack(map(np.ravel, g))
        return positions.swapaxes(0,1)

    dot5 = gen_grid(-0.5,0.5, 70) 
    dot10 = gen_grid(-1.0, 1.0, 50) 
    grid = np.concatenate((dot5,dot10))
    grid = torch.from_numpy(grid).float()
    grid = grid[ torch.randperm(grid.shape[0])[0:int(pc_size*query_per_point/3)] ]

    total_size = pc_size*query_per_point + grid.shape[0]
    #print("grid shape, total size: ",grid.shape, total_size)

    xyz = torch.empty(size=(total_size,3))
    gt_pt = torch.empty(size=(total_size,3))

    # sample xyz
    dists = torch.cdist(pc, pc)
    std, _ = torch.topk(dists, 50, dim=-1, largest=False)
    std = std[:,-1].unsqueeze(-1)

    # print("std: ",std.shape, std[0])
    # print("pc: ",pc.shape)
    # print("dists: ",dists.shape)

    count = 0
    for idx, p in enumerate(pc):
        # query locations from p
        q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                             size=(query_per_point, 3))

        # query locations in space
        q = p + q_loc
        xyz[count:count+query_per_point] = q
        count += query_per_point

    xyz[pc_size*query_per_point:] = grid 

    # nearest neighbor
    dists = torch.cdist(xyz, pc)
    _, min_idx = torch.min(dists, dim=-1)  
    gt_pt = pc[min_idx]
    return xyz.unsqueeze(0), gt_pt.unsqueeze(0)


def fast_opt(model, pc):
    num_iterations = 800
    xyz, gt_pt = fast_preprocess(pc)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=1e-4)

    # for n,p in model.named_parameters():
    #     init_param = p
    #     break
    #print("init param: ",init_param[0:10])

    #print("shapes: ", pc.shape, xyz.shape)
    for e in range(num_iterations):

        shape_vecs = model.encoder(pc, xyz)
        decoder_input = torch.cat([shape_vecs, xyz], dim=-1)
        pred_sdf = model.decoder(decoder_input).unsqueeze(-1)

        pc_vecs = model.encoder(pc, pc)
        pc_pred = model.decoder(torch.cat([pc_vecs, pc], dim=-1))

        pred_pt, gt_pt = model.get_unlab_offset(xyz, gt_pt, pred_sdf)

        # loss of pt offset and loss of L1
        unlabeled_loss = nn.MSELoss()(pred_pt, gt_pt)
        # using pc to supervise query as well
        pc_l1 = nn.L1Loss()(pc_pred, torch.zeros(*pc_pred.shape))

        loss = unlabeled_loss + 0.1*pc_l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e==num_iterations/2:
            for g in optimizer.param_groups:
                g['lr'] /= 2 

    # for n,p in model.named_parameters():
    #     final_param = p
    #     break
    #print("final param: ",final_param[0:10])

    return model





def get_unlab_offset(query_xyz, query_gt_pt, pred_sdf):
    dir_vec = F.normalize(query_xyz - query_gt_pt, dim=-1)

    # different for batch size=1 and batch_size >1
    # todo: combine, shouldn't need this condition
    if query_xyz.shape[0] ==1:
        pred_sdf = pred_sdf.unsqueeze(0)
        neg_idx = torch.where(pred_sdf.squeeze()<0)[0]
        pos_idx = torch.where(pred_sdf.squeeze()>=0)[0]

        neg_pred = query_xyz[:,neg_idx] + dir_vec[:, neg_idx] * pred_sdf[:,neg_idx]
        pos_pred = query_xyz[:,pos_idx] - dir_vec[:, pos_idx] * pred_sdf[:,pos_idx]

        pred_pt = torch.cat((neg_pred, pos_pred), dim=1)                                                                  
        query_gt_pt = torch.cat((query_gt_pt[:,neg_idx], query_gt_pt[:,pos_idx]), dim=1)
    
    else:
        # splits into a tuple of two tensors; one tensor for each dimension; then can use as index
        neg_idx = pred_sdf.squeeze()<0
        neg_idx = neg_idx.nonzero().split(1, dim=1) 

        pos_idx = pred_sdf.squeeze()>=0
        pos_idx = pos_idx.nonzero().split(1, dim=1)

        # based on sign of sdf value, need to direct in different direction
        # indexing in this way results in an extra dimension that should be squeezed
        neg_pred = query_xyz[neg_idx].squeeze(1) + dir_vec[neg_idx].squeeze(1) * pred_sdf[neg_idx].squeeze(1)
        pos_pred = query_xyz[pos_idx].squeeze(1) - dir_vec[pos_idx].squeeze(1) * pred_sdf[pos_idx].squeeze(1)

        # for batch size 4, query_per_batch 16384, 
        # dimension 4,16384,3 -> 4*16384, 3
        pred_pt = torch.cat((neg_pred, pos_pred), dim=0) # batches are combined
        query_gt_pt = torch.cat((query_gt_pt[neg_idx].squeeze(1), query_gt_pt[pos_idx].squeeze(1)), dim=0)

    return pred_pt, query_gt_pt


