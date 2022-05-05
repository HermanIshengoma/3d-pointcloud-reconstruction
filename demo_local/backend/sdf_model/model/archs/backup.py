#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import sys
from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear, MetaConv1d, MetaBatchNorm1d)
from torch.autograd import Variable
import numpy as np 

from collections import OrderedDict


class DeepSdfPointNet(MetaModule):
    def __init__(self, latent_size, samp_per_scene, reg_loss=False, sal_pn=True):
        super().__init__()
        self.samp_per_scene = samp_per_scene
        self.latent_size = latent_size
        self.input_size = latent_size+3
        self.reg_loss = reg_loss
        self.sal_pn = sal_pn
        
        # add weight norm later if this works
        # lin1 = nn.utils.weight_norm(lin1)
        self.lin1 = MetaLinear(self.input_size, 512)
        self.lin2 = MetaLinear(512, 512)
        self.lin3 = MetaLinear(512, 512)
        self.lin4 = MetaLinear(512, 512-self.input_size)
        # here in forward, concat with input to get correct dim
        self.lin5 = MetaLinear(512, 512)
        self.lin6 = MetaLinear(512, 512)
        self.lin7 = MetaLinear(512, 512)
        self.lin8 = MetaLinear(512, 512)
        # no normalization for final layer
        self.lin9 = MetaLinear(512, 1)
        
        # self.pointnet = MetaSequential( # remember to max pool in forward
        #     MetaConv1d(3, 64, kernel_size=1, bias=False), MetaBatchNorm1d(64), nn.ReLU(),
        #     MetaConv1d(64, 64, kernel_size=1, bias=False), MetaBatchNorm1d(64), nn.ReLU(),
        #     MetaConv1d(64, 64, kernel_size=1, bias=False), MetaBatchNorm1d(64), nn.ReLU(),
        #     MetaConv1d(64, 128, kernel_size=1, bias=False), MetaBatchNorm1d(128), nn.ReLU(),
        #     MetaConv1d(128, self.latent_size, kernel_size=1, bias=False), MetaBatchNorm1d(self.latent_size)
        #     )

        if self.sal_pn:
            self.pointnet = SalPointNet(c_dim=self.latent_size, hidden_dim=self.latent_size*2)
        else:
            self.pointnet = PointNetfeat(out_dim=self.latent_size, feature_transform=self.reg_loss)

    # x is a 3d coordinate and the pointcloud  
    # after processing them separately, concat to get the tensor for network training 
    def forward(self, x, params=None):

        coord, pc = x 
        # pc shape: (batch size, 1024, 3)
        
        if self.sal_pn:
            latent_mean, latent_std = self.pointnet(pc)
            latent_reg = 1.0e-3*(latent_mean.abs().mean(dim=-1) + (latent_std + 1).abs().mean(dim=-1))
            latent_reg = latent_reg.mean()
            dist = torch.distributions.Normal(latent_mean, torch.exp(latent_std))
            shape_vecs = dist.rsample()
            # print("l mean, std shape: ", latent_mean.shape, latent_std.shape)
            # print("l reg shape and [0]: ", latent_reg.shape, latent_reg[0])
            # print("shape_vecs shape: ",shape_vecs.shape)
        else:
            pc = pc.transpose(2,1).contiguous()
            shape_vecs, trans, trans_feat = self.pointnet(pc, params=self.get_subdict(params, 'pointnet'))
            latent_reg = self.pointnet.feature_transform_regularizer(trans_feat) * 0.001
        
        shape_vecs = shape_vecs.unsqueeze(2).repeat(1, 1, coord.shape[1]).transpose(2,1).contiguous()
        # print("shape_vecs shape: ",shape_vecs.shape)
        # print("coord shape: ", coord.shape)
        nn_input = torch.cat([shape_vecs, coord], dim=-1).float() 

        x = F.dropout(F.relu(self.lin1(nn_input, params=self.get_subdict(params, 'lin1'))), p=0.2)
        x = F.dropout(F.relu(self.lin2(x, params=self.get_subdict(params, 'lin2'))), p=0.2)
        x = F.dropout(F.relu(self.lin3(x, params=self.get_subdict(params, 'lin3'))), p=0.2)
        x = F.dropout(F.relu(self.lin4(x, params=self.get_subdict(params, 'lin4'))), p=0.2)
        x = torch.cat([x, nn_input], dim=-1) # concat input
        x = F.dropout(F.relu(self.lin5(x, params=self.get_subdict(params, 'lin5'))), p=0.2)
        x = F.dropout(F.relu(self.lin6(x, params=self.get_subdict(params, 'lin6'))), p=0.2)
        x = F.dropout(F.relu(self.lin7(x, params=self.get_subdict(params, 'lin7'))), p=0.2)
        x = F.dropout(F.relu(self.lin8(x, params=self.get_subdict(params, 'lin8'))), p=0.2)
        x = self.lin9(x, params=self.get_subdict(params, 'lin9'))

        if self.reg_loss:
            return torch.squeeze(nn.Tanh()(x)), latent_reg
        else:
            return torch.squeeze(nn.Tanh()(x))


class SalPointNet(nn.Module):
    ''' PointNet-based encoder network. Based on: https://github.com/autonomousvision/occupancy_networks
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=256, in_dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(in_dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        torch.nn.init.constant_(self.fc_mean.weight,0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = self.maxpool

    def forward(self, p):
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        c_mean = self.fc_mean(self.actvn(net))
        c_std = self.fc_std(self.actvn(net))

        return c_mean,c_std

    def maxpool(self, x, dim=-1, keepdim=False):
        out, _ = x.max(dim=dim, keepdim=keepdim)
        return out

        

class PointNetfeat(MetaModule):
    def __init__(self, out_dim=1024, global_feat = True, feature_transform = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.out_dim = out_dim
        self.conv1 = MetaConv1d(3, 64, 1)
        self.conv2 = MetaConv1d(64, 128, 1)
        self.conv3 = MetaConv1d(128, self.out_dim, 1)
        self.bn1 = MetaBatchNorm1d(64)
        self.bn2 = MetaBatchNorm1d(128)
        self.bn3 = MetaBatchNorm1d(self.out_dim)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, params=None):
        n_pts = x.size()[2]
        trans = self.stn(x, params=self.get_subdict(params, 'stn'))
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x, params=self.get_subdict(params, 'conv1')), params=self.get_subdict(params, 'bn1')))

        if self.feature_transform:
            trans_feat = self.fstn(x, params=self.get_subdict(params, 'fstn'))
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x, params=self.get_subdict(params, 'conv2')), params=self.get_subdict(params, 'bn2')))
        x = self.bn3(self.conv3(x, params=self.get_subdict(params, 'conv3')), params=self.get_subdict(params, 'bn3'))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.out_dim, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

    def feature_transform_regularizer(self, trans):
        d = trans.size()[1]
        batchsize = trans.size()[0]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
        return loss


class STNkd(MetaModule):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = MetaConv1d(k, 64, 1)
        self.conv2 = MetaConv1d(64, 128, 1)
        self.conv3 = MetaConv1d(128, 1024, 1)
        self.fc1 = MetaLinear(1024, 512)
        self.fc2 = MetaLinear(512, 256)
        self.fc3 = MetaLinear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = MetaBatchNorm1d(64)
        self.bn2 = MetaBatchNorm1d(128)
        self.bn3 = MetaBatchNorm1d(1024)
        self.bn4 = MetaBatchNorm1d(512)
        self.bn5 = MetaBatchNorm1d(256)

        self.k = k

    def forward(self, x, params=None):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x, params=self.get_subdict(params, 'conv1')), params=self.get_subdict(params, 'bn1')))
        x = F.relu(self.bn2(self.conv2(x, params=self.get_subdict(params, 'conv2')), params=self.get_subdict(params, 'bn2')))
        x = F.relu(self.bn3(self.conv3(x, params=self.get_subdict(params, 'conv3')), params=self.get_subdict(params, 'bn3')))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x, params=self.get_subdict(params, 'fc1')), params=self.get_subdict(params, 'bn4')))
        x = F.relu(self.bn5(self.fc2(x, params=self.get_subdict(params, 'fc2')), params=self.get_subdict(params, 'bn5')))
        x = self.fc3(x, params=self.get_subdict(params, 'fc3'))

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class STN3d(MetaModule):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = MetaConv1d(3, 64, 1)
        self.conv2 = MetaConv1d(64, 128, 1)
        self.conv3 = MetaConv1d(128, 1024, 1)
        self.fc1 = MetaLinear(1024, 512)
        self.fc2 = MetaLinear(512, 256)
        self.fc3 = MetaLinear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = MetaBatchNorm1d(64)
        self.bn2 = MetaBatchNorm1d(128)
        self.bn3 = MetaBatchNorm1d(1024)
        self.bn4 = MetaBatchNorm1d(512)
        self.bn5 = MetaBatchNorm1d(256)


    def forward(self, x, params=None):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x, params=self.get_subdict(params, 'conv1')), params=self.get_subdict(params, 'bn1')))
        x = F.relu(self.bn2(self.conv2(x, params=self.get_subdict(params, 'conv2')), params=self.get_subdict(params, 'bn2')))
        x = F.relu(self.bn3(self.conv3(x, params=self.get_subdict(params, 'conv3')), params=self.get_subdict(params, 'bn3')))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x, params=self.get_subdict(params, 'fc1')), params=self.get_subdict(params, 'bn4')))
        x = F.relu(self.bn5(self.fc2(x, params=self.get_subdict(params, 'fc2')), params=self.get_subdict(params, 'bn5')))
        x = self.fc3(x, params=self.get_subdict(params, 'fc3'))

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
