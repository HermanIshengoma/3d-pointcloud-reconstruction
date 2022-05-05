#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh
from scipy.spatial import cKDTree as KDTree
from . import metrics
import csv
#from pytorch3d.loss import chamfer_distance

def main(gt_pc, recon_mesh, out_file, mesh_name):

    gt_pc = gt_pc.cpu().detach().numpy().squeeze()

    #print(os.getcwd()) # root/SDF-lib
    # recon_mesh: config_ckpt/....reconstruct
    recon_mesh = trimesh.load(os.path.join(os.getcwd(), recon_mesh)+".ply")
    recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, gt_pc.shape[0])

    #print("gt pc shape: ",gt_pc.shape)
    #print("recon_pc shape: ", recon_pc.shape)

    recon_kd_tree = KDTree(recon_pc)
    one_distances, one_vertex_ids = recon_kd_tree.query(gt_pc)
    gt_to_recon_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_kd_tree = KDTree(gt_pc)
    two_distances, two_vertex_ids = gt_kd_tree.query(recon_pc)
    recon_to_gt_chamfer = np.mean(np.square(two_distances))
    
    loss_chamfer = gt_to_recon_chamfer + recon_to_gt_chamfer    

    #loss_chamfer, _ = chamfer_distance(gt_pc, recon_pc)

    out_file = os.path.join(os.getcwd(), out_file)
    #print("outfile: ",out_file)
    #Path(out_file).mkdir(parents=True, exist_ok=True)

    with open(out_file,"a",) as f:
        writer = csv.writer(f)
        writer.writerow([mesh_name[0],loss_chamfer])

def out(gt_pc, recon_mesh):

    gt_pc = gt_pc.squeeze()

    recon_mesh = trimesh.load(recon_mesh)
    recon_pc, _ = trimesh.sample.sample_surface(recon_mesh, gt_pc.shape[0])

    #print("gt pc shape: ",gt_pc.shape)
    #print("recon_pc shape: ", recon_pc.shape)

    recon_kd_tree = KDTree(recon_pc)
    one_distances, one_vertex_ids = recon_kd_tree.query(gt_pc)
    gt_to_recon_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_kd_tree = KDTree(gt_pc)
    two_distances, two_vertex_ids = gt_kd_tree.query(recon_pc)
    recon_to_gt_chamfer = np.mean(np.square(two_distances))
    
    loss_chamfer = gt_to_recon_chamfer + recon_to_gt_chamfer    
    return round(loss_chamfer, 5)



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to evaluate.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    evaluate(
        args.experiment_directory,
        args.checkpoint,
        args.data_source,
        args.split_filename,
    )
