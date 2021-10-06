from mylib import random_walk_torch, downsample_query_ball
import torch
import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstraction, farthest_point_sample, index_points, sample_and_group
import numpy as np
import torch.nn.functional as F
from utils import group_points_2, group_points
import open3d as o3d

class UnpoolObj(nn.Module):
    def __init__(self):
        super(UnpoolObj, self).__init__()
    
    def forward(self, xyz, points):
        return random_walk_torch(xyz.squeeze(-1).permute(0, 2, 1), 32, 0.05), random_walk_torch(points.squeeze(-1).permute(0, 2, 1), 32, 0.0)

class PoolObj(nn.Module):
    def __init__(self):
        super(PoolObj, self).__init__()
    
    def forward(self, xyz, points):
        _xyz = xyz.squeeze(-1).permute(0, 2, 1)
        _points = points.squeeze(-1).permute(0, 2, 1)

        B, N, C = _xyz.shape
        # idx = farthest_point_sample(_xyz, N//2)
        idx = downsample_query_ball(_xyz, N//2, 0.05, 0.02)
        return index_points(_xyz, idx), index_points(_points, idx)

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.netR = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(3, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1, 64),stride=1),
            # B*128*sample_num_level1*1
        )

        self.fc = nn.Sequential(
            nn.Linear(128*21, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 63)
        )
    
    def forward(self, xyz):
        B, N = xyz.shape
        joints = xyz.reshape(-1, 21, 3)
        # joints = joints.permute(0, 2, 1)
        joints = random_walk_torch(joints, 64, 0.02)
        joints, joints_center = group_points_2(joints.permute(0, 2, 1), 21*64, 21, 64, 0.05)

        x = self.netR(joints)
        x = x.view(-1, 128*21)
        x = self.fc(x)

        return x

nstates_plus_1 = [64,64,128]
nstates_plus_2 = [128,128,256]
nstates_plus_3 = [256,512,1024,1024,512]
class PointUNetObj(nn.Module):
    def __init__(self, ball_radius2):
        super(PointUNetObj, self).__init__()
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.04, nsample=64, mlp=[64, 64, 128], in_channel=6, group_all=False) # 512x32
        # # self.sa5 = PointNetSetAbstraction(npoint=128, radius=0.1, nsample=64, mlp=[128, 128, 256], in_channel=128+3, group_all=False) # 128x32
        # self.pool = PoolObj() # 256x128
        # self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.1, nsample=64, mlp=[128, 128, 256], in_channel=128+3, group_all=False) # 32x32
        # self.unpool = UnpoolObj() # 896x32
        # # self.sa3 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, mlp=[256, 256, 512], in_channel=256+3, group_all=False)
        # # self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, mlp=[512, 512, 1024], in_channel=512+3, group_all=True)
        # self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512], in_channel=256+3, group_all=True)
        # # self.fc1 = nn.Linear(1024, 512)
        # # self.bn1 = nn.BatchNorm1d(512)
        # # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, 42)

        self.num_outputs = 87
        self.knn_K = 64
        self.ball_radius2 = ball_radius2
        self.sample_num_level1 = 512
        self.sample_num_level2 = 32
        self.INPUT_FEATURE_NUM = 6
        
        self.netR_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*128*sample_num_level1*1
        )
        
        self.netR_2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )
        
        self.netR_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((1152,1),stride=1),
            # B*1024*1*1
        )
        
        self.netR_FC = nn.Sequential(
            # B*1024
            nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
        )

        self.pool = PoolObj() # 256x128
        self.unpool = UnpoolObj()
        self.refine_net = RefineNet()
    
    # def forward(self, xyz):
    #     B, _, _ = xyz.shape

    #     l0_xyz = xyz[:, :, :3]
    #     l0_points = xyz[:, :, 3:]

    #     # print(l0_xyz[0].cpu().numpy()[:10])
    #     # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(l0_xyz[0].cpu().numpy()))
    #     # o3d.visualization.draw_geometries([pcd])

    #     l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1))
    #     l5_xyz, l5_points = self.sa5(l1_xyz, l1_points)

    #     l2_xyz_inp, l2_points_inp = self.pool(l1_xyz, l1_points)
    #     l2_xyz, l2_points = self.sa2(l2_xyz_inp.permute(0, 2, 1), l2_points_inp.permute(0, 2, 1))

    #     l3_xyz_inp, l3_points_inp = self.unpool(l2_xyz, l2_points)

    #     # print(l3_points_inp.shape)
    #     # print(l3_points_inp[0][:10])

    #     l3_xyz_inp = torch.cat((l5_xyz, l3_xyz_inp.permute(0, 2, 1)), dim=2)
    #     l3_points_inp = torch.cat((l5_points, l3_points_inp.permute(0, 2, 1)), dim=2)
    #     # l3_xyz, l3_points = self.sa3(l3_xyz_inp, l3_points_inp)

    #     # print(l3_xyz.permute(0, 2, 1)[0].cpu().numpy()[:10])
    #     # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(l3_xyz.permute(0, 2, 1)[0].cpu().numpy()))
    #     # o3d.visualization.draw_geometries([pcd])

    #     # l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
    #     # out = l4_points.view(B, 1024)

    #     l4_xyz, l4_points = self.sa4(l3_xyz_inp, l3_points_inp)
    #     out = l4_points.view(B, 512)

    #     # out = self.drop1(F.relu(self.bn1(self.fc1(out))))
    #     out = self.drop2(F.relu(self.bn2(self.fc2(out))))
    #     out = self.fc3(out)

    #     return out
    def forward(self, x, y):
        B, _, _, _ = x.shape

        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        x = self.netR_1(x)
        # B*128*sample_num_level1*1
        x_ = torch.cat((y, x),1).squeeze(-1)
        # B*(3+128)*sample_num_level1

        
        # inputs_level2_, inputs_level2_center_ = group_points_2(x_, self.sample_num_level1, 128, self.knn_K, self.ball_radius2)
        x_ = x.squeeze_(-1).permute(0, 2, 1)
        y_ = y.squeeze_(-1).permute(0, 2, 1)
        inputs_level2_center_, inputs_level2_ = sample_and_group(128, self.ball_radius2, self.knn_K, y_, x_)
        inputs_level2_ = inputs_level2_.permute(0, 3, 1, 2)
        inputs_level2_center_ = inputs_level2_center_.permute(0, 2, 1).unsqueeze_(3)
        x_ = self.netR_2(inputs_level2_)

        y_pool, x_pool = self.pool(y, x)
        x = torch.cat((y_pool.permute(0, 2, 1), x_pool.permute(0, 2, 1)),1)
        
        # inputs_level2, inputs_level2_center = group_points_2(x, self.sample_num_level1 // 2, self.sample_num_level2, self.knn_K, self.ball_radius2)
        inputs_level2_center, inputs_level2 = sample_and_group(self.sample_num_level2, self.ball_radius2, self.knn_K, y_pool, x_pool)
        inputs_level2 = inputs_level2.permute(0, 3, 1, 2)
        inputs_level2_center = inputs_level2_center.permute(0, 2, 1).unsqueeze_(3)
        
        # B*131*sample_num_level2*knn_K
        x = self.netR_2(inputs_level2)
        # B*256*sample_num_level2*1
        # x = torch.cat((inputs_level2_center, x),1)
        # B*259*sample_num_level2*1

        y_unpool, x_unpool = self.unpool(inputs_level2_center, x)
        # print(y_unpool.shape, x_unpool.shape)

        x = x_pool.permute(0, 2, 1)
        x = torch.cat((y_unpool.permute(0, 2, 1), x_unpool.permute(0, 2, 1)), 1).unsqueeze(3)

        x_ = torch.cat((inputs_level2_center_, x_), 1)
        x = torch.cat((x, x_), 2)

        x = self.netR_3(x)

        # print(x.shape)

        # B*1024*1*1
        
        x = x.view(-1,nstates_plus_3[2])

        # print(x.shape)
        # B*1024

        x = self.netR_FC(x)
        # B*num_outputs

        # x = self.refine_net(x)

        return x