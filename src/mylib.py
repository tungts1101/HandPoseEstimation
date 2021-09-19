import time
from scipy.spatial import distance
import numpy as np
import random
import torch

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
        xyz: pointcloud centroids, [B, N, 3]
        npoint: number of samples
    Return:
        points: upsampled pointcloud index, [B, N*k, 3]
    """
    device = xyz.device
    B, N, C = xyz.shape
    points = torch.zeros(B, N*k, C, dtype=torch.long).to(device)
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