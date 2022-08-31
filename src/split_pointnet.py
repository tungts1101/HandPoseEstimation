import torch
import torch.nn as nn
import math
from utils import group_points_2

nstates_plus_1 = [32,32,64]
nstates_plus_2 = [64,128,128]
nstates_plus_3 = [128,256,512,512,256]

def split(points):
    result = []

    device = points.device
    B, N, C = points.shape
    xyz = points[:, :, :3]

    center = torch.div(torch.sum(xyz, dim=1), N)
    centers = []
    centers.append(center - torch.tensor([-0.1, 0.5, 0.0]).to(device))         # thumb
    centers.append(center - torch.tensor([0.3, 0.3, 0.0]).to(device))          # index
    centers.append(center - torch.tensor([0.35, 0.0, 0.0]).to(device))         # middle
    centers.append(center - torch.tensor([0.3, -0.3, 0.0]).to(device))         # ring
    centers.append(center - torch.tensor([0.25, -0.5, 0.0]).to(device))        # pinky
    centers.append(center - torch.tensor([-0.25, -0.25, -0.25]).to(device))    # palm

    num_points = [256, 256, 256, 256, 256, 512]                     # B*256*C for each finger and B*512*C for palm area
    for i in range(len(num_points)):
        result.append(torch.zeros([B,num_points[i],C]).to(device))

    for i in range(len(result)):
        dist = torch.norm(xyz - centers[i].unsqueeze_(-1).permute(0, 2, 1), dim=2, p=None)
        knn = torch.topk(dist, num_points[i], largest=False)
        result[i] = torch.index_select(points, dim=1, index=knn.indices[0])

    return result 

class SplitPointNet(nn.Module):
    def __init__(self, ball_radius1, ball_radius2):
        super(SplitPointNet, self).__init__()
        self.knn_K = 64
        self.ball_radius1 = ball_radius1
        self.ball_radius2 = ball_radius2
        self.sample_num_level1 = [64, 64, 64, 64, 64, 128]
        self.sample_num_level2 = [32, 32, 32, 32, 32, 64]
        self.INPUT_FEATURE_NUM = 6
        self.num_outputs       = [12, 12, 12, 12, 12, 18]

        self.net1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[1], kernel_size=(1, 1)),
                nn.BatchNorm2d(nstates_plus_1[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
                nn.BatchNorm2d(nstates_plus_1[2]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((1,self.knn_K),stride=1)
            ) for _ in range(6)
        ])

        self.net2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
                nn.BatchNorm2d(nstates_plus_2[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
                nn.BatchNorm2d(nstates_plus_2[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((1,self.knn_K),stride=1)
            ) for _ in range(6)
        ])

        self.net3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
                nn.BatchNorm2d(nstates_plus_3[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
                nn.BatchNorm2d(nstates_plus_3[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
                nn.BatchNorm2d(nstates_plus_3[2]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((self.sample_num_level2[i],1),stride=1),
            ) for i in range(6)
        ])
        
        # # 1 PSA
        # self.net1 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_2[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[0]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[1]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((1,self.knn_K),stride=1)
        #     ) for _ in range(6)
        # ])

        # self.net3 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[0]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[2]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((self.sample_num_level2[i],1),stride=1),
        #     ) for i in range(6)
        # ])

        # # 1 conv
        # self.net1 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_1[0]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((1,self.knn_K),stride=1)
        #     ) for _ in range(6)
        # ])

        # self.net2 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(3+nstates_plus_1[0], nstates_plus_2[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[0]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((1,self.knn_K),stride=1)
        #     ) for _ in range(6)
        # ])

        # self.net3 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(3+nstates_plus_2[0], nstates_plus_3[2], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[2]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((self.sample_num_level2[i],1),stride=1),
        #     ) for i in range(6)
        # ])

        # # 5 conv
        # self.net1 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_1[0]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_1[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_1[1], nstates_plus_1[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_1[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_1[1], nstates_plus_1[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_1[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_1[2]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((1,self.knn_K),stride=1)
        #     ) for _ in range(6)
        # ])

        # self.net2 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[0]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_2[1], nstates_plus_2[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_2[1], nstates_plus_2[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_2[2]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((1,self.knn_K),stride=1)
        #     ) for _ in range(6)
        # ])

        # self.net3 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[0]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_3[0], nstates_plus_3[0], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[0]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_3[1], nstates_plus_3[1], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[1]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[2]),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(nstates_plus_3[2], nstates_plus_3[2], kernel_size=(1, 1)),
        #         nn.BatchNorm2d(nstates_plus_3[2]),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d((self.sample_num_level2[i],1),stride=1),
        #     ) for i in range(6)
        # ])

        self.netFC = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
                nn.BatchNorm1d(nstates_plus_3[4]),
                nn.ReLU(inplace=True),
                nn.Linear(nstates_plus_3[4], self.num_outputs[i]),
            ) for i in range(6)
        ])

    def forward(self, points):
        device = points.device
        B, _, _ = points.shape
        result = torch.zeros([B,63]).to(device)

        parts = split(points)
        for index, part in enumerate(parts):
            inputs_level1, inputs_level1_center = group_points_2(
                part.permute(0, 2, 1), part.shape[1], self.sample_num_level1[index], self.knn_K, self.ball_radius1)
            x = self.net1[index](inputs_level1)
            x = torch.cat((inputs_level1_center, x), 1).squeeze(-1)

            inputs_level2, inputs_level2_center = group_points_2(
                x, self.sample_num_level1[index], self.sample_num_level2[index], self.knn_K, self.ball_radius2)
            x = self.net2[index](inputs_level2)
            x = torch.cat((inputs_level2_center, x), 1)
            
            # # 1 PSA
            # inputs_level1, inputs_level1_center = group_points_2(
            #     part.permute(0, 2, 1), part.shape[1], self.sample_num_level2[index], self.knn_K, self.ball_radius2)
            # x = self.net1[index](inputs_level1)
            # x = torch.cat((inputs_level1_center, x), 1)

            x = self.net3[index](x)                 # 64*512*1*1
            x = x.view(-1,nstates_plus_3[2])        # 64*512
            x = self.netFC[index](x)

            if index == 5: # palm area
                result[:, :3] = x[:, :3]
                for i in range(5):
                    result[:, 3+i*3:3+i*3+3] += 0.8 * x[:, 3+i*3:3+i*3+3]
            else:
                result[:, (index+1)*3:(index+1)*3+3] += 0.2 * x[:, :3]
                for i in range(3):
                    result[:, 18+index*9+i*3:18+index*9+i*3+3] = x[:, 3+i*3:3+i*3+3]

        return result