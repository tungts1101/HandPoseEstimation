from mylib import random_walk_torch
import torch
import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, farthest_point_sample, index_points
import numpy as np
import torch.nn.functional as F

class UnpoolObj(nn.Module):
    def __init__(self):
        super(UnpoolObj, self).__init__()
    
    def forward(self, xyz, points):
        return random_walk_torch(xyz.permute(0, 2, 1), 16), random_walk_torch(points.permute(0, 2, 1), 16)

class PoolObj(nn.Module):
    def __init__(self):
        super(PoolObj, self).__init__()
    
    def forward(self, xyz, points):
        _xyz = xyz.permute(0, 2, 1)
        _points = points.permute(0, 2, 1)

        B, N, C = _xyz.shape
        idx = farthest_point_sample(_xyz, N//2)
        return index_points(_xyz, idx), index_points(_points, idx)

class PointUNetObj(nn.Module):
    def __init__(self):
        super(PointUNetObj, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, mlp=[16, 16, 32], in_channel=6, group_all=False) # 512x32
        self.pool = PoolObj() # 128x32
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.5, nsample=64, mlp=[32, 32, 32], in_channel=32+3, group_all=False) # 32x32
        self.unpool = UnpoolObj() # 512x32
        # self.sa3 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=64, mlp=[32, 32, 64], in_channel=32+3, group_all=True)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, mlp=[128, 256, 512], in_channel=32+3, group_all=True)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 42)

    
    def forward(self, xyz):
        B, _, _ = xyz.shape

        l0_xyz = xyz[:, :, :3]
        l0_points = xyz[:, :, 3:]
        l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1))

        l2_xyz_inp, l2_points_inp = self.pool(l1_xyz, l1_points)
        l2_xyz, l2_points = self.sa2(l2_xyz_inp.permute(0, 2, 1), l2_points_inp.permute(0, 2, 1))

        l3_xyz_inp, l3_points_inp = self.unpool(l2_xyz, l2_points)
        l3_xyz_inp = torch.cat((l1_xyz, l3_xyz_inp.permute(0, 2, 1)), dim=2)
        l3_points_inp = torch.cat((l1_points, l3_points_inp.permute(0, 2, 1)), dim=2)
        l3_xyz, l3_points = self.sa3(l3_xyz_inp, l3_points_inp)
        out = l3_points.view(B, 512)
        out = self.drop1(F.relu(self.bn1(self.fc1(out))))
        out = self.fc2(out)

        return out