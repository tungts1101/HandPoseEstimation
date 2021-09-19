from mylib import random_walk_torch
import torch
import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, farthest_point_sample, index_points
import numpy as np
import torch.nn.functional as F
from utils import group_points_2
import open3d as o3d
class UnpoolObj(nn.Module):
    def __init__(self):
        super(UnpoolObj, self).__init__()
    
    def forward(self, xyz, points):
        return random_walk_torch(xyz.permute(0, 2, 1), 64, 0.05), random_walk_torch(points.permute(0, 2, 1), 64, 0.0)

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
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.04, nsample=64, mlp=[64, 64, 128], in_channel=6, group_all=False) # 512x32
        self.sa5 = PointNetSetAbstraction(npoint=128, radius=0.1, nsample=64, mlp=[128, 128, 256], in_channel=128+3, group_all=False) # 128x32
        self.pool = PoolObj() # 256x128
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.1, nsample=64, mlp=[128, 128, 256], in_channel=128+3, group_all=False) # 32x32
        self.unpool = UnpoolObj() # 896x32
        self.sa3 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, mlp=[256, 256, 512], in_channel=256+3, group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, mlp=[512, 512, 1024], in_channel=512+3, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 42)
    
    def forward(self, xyz):
        B, _, _ = xyz.shape

        l0_xyz = xyz[:, :, :3]
        l0_points = xyz[:, :, 3:]

        # print(l0_xyz[0].cpu().numpy()[:10])
        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(l0_xyz[0].cpu().numpy()))
        # o3d.visualization.draw_geometries([pcd])

        l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1))
        l5_xyz, l5_points = self.sa5(l1_xyz, l1_points)

        l2_xyz_inp, l2_points_inp = self.pool(l1_xyz, l1_points)
        l2_xyz, l2_points = self.sa2(l2_xyz_inp.permute(0, 2, 1), l2_points_inp.permute(0, 2, 1))

        l3_xyz_inp, l3_points_inp = self.unpool(l2_xyz, l2_points)

        # print(l3_points_inp.shape)
        # print(l3_points_inp[0][:10])

        l3_xyz_inp = torch.cat((l5_xyz, l3_xyz_inp.permute(0, 2, 1)), dim=2)
        l3_points_inp = torch.cat((l5_points, l3_points_inp.permute(0, 2, 1)), dim=2)
        l3_xyz, l3_points = self.sa3(l3_xyz_inp, l3_points_inp)

        # print(l3_xyz.permute(0, 2, 1)[0].cpu().numpy()[:10])
        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(l3_xyz.permute(0, 2, 1)[0].cpu().numpy()))
        # o3d.visualization.draw_geometries([pcd])

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # l4_xyz, l4_points = self.sa4(l3_xyz_inp, l3_points_inp)
        out = l4_points.view(B, 1024)

        out = self.drop1(F.relu(self.bn1(self.fc1(out))))
        out = self.drop2(F.relu(self.bn2(self.fc2(out))))
        out = self.fc3(out)

        return out