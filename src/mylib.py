import time
from scipy.spatial import distance
import numpy as np
import random
import torch
import torch.nn as nn

def random_walk(center, k=10, max_dis=0.5):
    points = np.zeros((k, 3), dtype=np.float32)

    for i in range(k):
        point = center + max_dis * (np.random.random_sample((3,)) - 1.0)
        points[i] = point

    return points

def knn_search(xyz, k=10):
    D = distance.squareform(distance.pdist(xyz))
    closest = np.argsort(D, axis=1)
    return closest[:,1:k+1]

def farthest_point_sampling(pts, k):
    # farthest point sampling
    def cal_dis(p0, points):
        return ((p0 - points)**2).sum(axis=1)

    if len(pts) < k:
        return [i for i in range(len(pts))] + [np.random.randint(len(pts)) for _ in range(k - len(pts))]

    indices = np.zeros((k, ), dtype=np.uint32)
    indices[0] = np.random.randint(len(pts))
    min_distances = cal_dis(pts[indices[0]], pts)
    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        min_distances = np.minimum(min_distances, cal_dis(pts[indices[i]], pts))
    return indices

def random_walk_torch(xyz, k=10, max_dis=0.1):
    """
    Input:
        xyz: pointcloud centroids, [B, N, C]
        npoint: number of samples
    Return:
        points: upsampled pointcloud index, [B, N*k, C]
    """
    device = xyz.device
    B, N, C = xyz.shape
    points = torch.zeros(B, N*k, C, dtype=torch.float32).to(device)
    for i_batch in range(B):
        for i_center in range(N):
            center = xyz[i_batch, i_center, :]
            for i in range(k):
                point = center + max_dis * (torch.rand(C,).to(device) - 1.0)
                points[i_batch, i_center*k + i:, :] = point

    return points

def timeit(tag, start_time):
    end_time = time.time()
    print("{}: {}s".format(tag, end_time - start_time))
    return end_time

# class PointNetSetAbstraction(nn.Module):
#     def __init__(self, inp_channels, out_channels, mlp, group_all):
#         super(PointNetSetAbstraction, self).__init__()

#         self.netR_1 = nn.Sequential(
#             # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
#             nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
#             nn.BatchNorm2d(nstates_plus_1[0]),
#             nn.ReLU(inplace=True),
#             # B*64*sample_num_level1*knn_K
#             nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
#             nn.BatchNorm2d(nstates_plus_1[1]),
#             nn.ReLU(inplace=True),
#             # B*64*sample_num_level1*knn_K
#             nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
#             nn.BatchNorm2d(nstates_plus_1[2]),
#             nn.ReLU(inplace=True),
#             # B*128*sample_num_level1*knn_K
#             nn.MaxPool2d((1,self.knn_K),stride=1)
#             # B*128*sample_num_level1*1
#         )
    
#     def forward(self, x, y):

