import torch
import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction
import numpy as np
import torch.nn.functional as F

pointset_level_1 = [64, 64, 128]
pointset_level_2 = [128, 128, 256]
pointset_level_3 = [256, 512, 1024]
class NetworkObj(nn.Module):
    def __init__(self):
        super(NetworkObj, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, mlp=pointset_level_1, in_channel=6, group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.5, nsample=64, mlp=pointset_level_2, in_channel=pointset_level_1[2] + 3, group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, mlp=pointset_level_3, in_channel=pointset_level_2[2] + 3, group_all=True)
        self.fc1 = nn.Linear(pointset_level_3[2], 512)
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
        l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1))
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.shape, l3_points.shape)
        out = l3_points.view(B, 1024)
        out = self.drop1(F.relu(self.bn1(self.fc1(out))))
        out = self.drop2(F.relu(self.bn2(self.fc2(out))))
        out = self.fc3(out)
        return out