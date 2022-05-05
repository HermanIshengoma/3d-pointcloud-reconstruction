#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import math
import numpy as np
import plyfile
import skimage.measure
import time
import torch

#import deep_sdf.utils

voxel_origin = [-1, -1, -1]


def create_grid(N):
    voxel_size = 2.0 / (N - 1)
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    return samples 


def create_mesh_single(
    decoder, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    
    #print("resolution: ",N)    
    start = time.time()
    ply_filename = filename

    decoder.eval()

    samples = create_grid(N)
    voxel_size = 2.0 / (N - 1)
    
    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3]
        sample_subset = sample_subset.reshape(-1,3)

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder.decoder( sample_subset ).squeeze().detach().cpu()
        )
        head += max_batch
    
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    #print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
    )

def create_mesh_prior(
    decoder, pc, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    
    #print("resolution: ",N)    
    start = time.time()
    ply_filename = filename

    decoder.eval()

    samples = create_grid(N)
    voxel_size = 2.0 / (N - 1)
    
    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3]
        sample_subset = sample_subset.reshape(1, -1,3).float()

        shape_vecs = decoder.encoder(pc, sample_subset)
        decoder_input = torch.cat([shape_vecs, sample_subset], dim=-1)

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder.decoder( decoder_input ).squeeze().detach().cpu()
        )
        head += max_batch
    
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    #print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()
    #print("input shape: ", pytorch_3d_sdf_tensor.shape)
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    #print("input[0:10]: ",pytorch_3d_sdf_tensor[0:10])  
    #print("input min max: ", numpy_3d_sdf_tensor.min(), numpy_3d_sdf_tensor.max())

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print(e)
        print("skipping {}".format(ply_filename_out))
        return
    #print("marching output shape: ", verts.shape, faces.shape, normals.shape, values.shape)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

