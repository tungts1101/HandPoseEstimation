import torch
import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction
import numpy as np
import torch.nn.functional as F
import utils

pointset_level_1 = [64, 64, 128]
pointset_level_2 = [128, 128, 256]
pointset_level_3 = [64, 64, 128]
class CascadedNetworkObj(nn.Module):
    def __init__(self):
        super(CascadedNetworkObj, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, mlp=pointset_level_1, in_channel=6, group_all=False)
        self.stage1 = [
            nn.Linear(pointset_level_1[2] * 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,42)
        ]
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.5, nsample=64, mlp=pointset_level_2, in_channel=pointset_level_1[2] + 3, group_all=False)
        self.stage2 = [
            nn.Linear(pointset_level_2[2] * 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,42)
        ]
        self.sa3 = PointNetSetAbstraction(npoint=512, radius=1.0, nsample=64, mlp=pointset_level_3, in_channel=6, group_all=False)
        self.stage3 = [
            nn.Linear(pointset_level_3[2] * 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 42)
        ]
        self.stage1 = nn.ModuleList(self.stage1)
        self.stage2 = nn.ModuleList(self.stage2)
        self.stage3 = nn.ModuleList(self.stage3)
    
    def forward(self, xyz, pca_mean, pca_coeff):
        B, _, _ = xyz.shape

        l0_xyz = xyz[:, :, :3]
        l0_points = xyz[:, :, 3:]

        l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1))
        estimate_stage_1 = l1_points.view(B, -1)
        for net in self.stage1:
            estimate_stage_1 = net(estimate_stage_1)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        estimate_stage_2 = l2_points.view(B, -1)
        for net in self.stage2:
            estimate_stage_2 = net(estimate_stage_2)

        pca_mean = pca_mean.expand(estimate_stage_2.data.size(0), pca_mean.size(1))
        estimate_joints = torch.addmm(pca_mean, estimate_stage_2.data, pca_coeff)
        input_sa3 = utils.group_points_around_sample(xyz, estimate_joints)

        input_sa3_xyz = input_sa3[:, :, :3]
        input_sa3_points = input_sa3[:, :, 3:]

        l3_xyz, l3_points = self.sa3(input_sa3_xyz.permute(0, 2, 1), input_sa3_points.permute(0, 2, 1))
        estimate_stage_3 = l3_points.view(B, -1)
        for net in self.stage1:
            estimate_stage_3 = net(estimate_stage_3)

        return estimate_stage_1, estimate_stage_2, estimate_stage_3