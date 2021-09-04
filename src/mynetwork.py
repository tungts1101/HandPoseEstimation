import torch
import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction
import numpy as np

class NetworkObj(nn.Module):
    def __init__(self,normal_channel=True):
        super(NetworkObj, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], in_channel=in_channel, group_all=False)

    
    def forward(self, xyz):
        return torch.from_numpy(np.random.random_sample((xyz.shape[0], 42)).astype(np.float32)).to(torch.device('cuda:0'))
        # B, _, _ = xyz.shape
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        #     norm = None
        # l1_xyz, l1_points = self.sa1(xyz, None)

        # print(l1_xyz.shape)
        # print(l1_points.shape)