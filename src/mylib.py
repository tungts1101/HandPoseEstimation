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

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

# def query_points(xyz, k=10):
#     """
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     points = torch.zeros(B, k, C, dtype=torch.float32).to(device)
#     sqrdists = square_distance(xyz, xyz)
#     # group_idx[sqrdists > radius ** 2] = N
#     # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]


def timeit(tag, start_time):
    end_time = time.time()
    print("{}: {}s".format(tag, end_time - start_time))
    return end_time

def downsample_random_walk(xyz, knn, num_sample, num_loop, prob):
    B, N, C = xyz.shape

    sqrdist = square_distance(xyz, xyz)
    around_indices = torch.argsort(sqrdist, dim=1)[:, :, :knn]    
    point_rank = [[0 for _ in range(N)] for _ in range(B)]

    for i_batch in range(B):
        idx = np.random.randint(N)
        point_rank[i_batch][idx] += 1

        for _ in range(num_loop):
            if np.random.rand() < prob:
                idx = np.random.randint(N)
            else:
                nb_idx = np.random.randint(len(around_indices[i_batch, idx]))
                idx = around_indices[i_batch, idx][nb_idx]

            point_rank[i_batch][idx] += 1

    point_idx = torch.tensor(np.argsort(point_rank)[:, :num_sample])
    return point_idx

def downsample_query_ball(xyz, num_sample, radius1, radius2):
    sqrdist = square_distance(xyz, xyz)

    device = xyz.device
    B, N, C = xyz.shape

    mask = torch.ones(B, N).to(device)
    mask = 1 - torch.diag_embed(mask)

    nb_rad_1 = ((sqrdist <= radius1)*mask).to(device)

    num_nb_rad_1 = torch.count_nonzero(nb_rad_1, dim=1).to(device)
    num_nb_rad_2 = torch.count_nonzero((sqrdist <= radius2)*mask, dim=1).unsqueeze_(2).type(torch.FloatTensor).to(device)

    div = torch.sum(torch.bmm(nb_rad_1, num_nb_rad_2), dim=2).to(device)
    point_rank = num_nb_rad_1 / (div + 1e-6)
    point_idx = torch.argsort(point_rank, dim=1)[:, :num_sample]
    return point_idx

class PNSA(nn.Module):
    def __init__(self, inp_channels, out_channels, mlp, group_all, knn_k):
        super(PNSA, self).__init__()

        layers = []
        layers.append(nn.Conv2d(inp_channels, mlp[0], kernel_size=(1,1)))
        layers.append(nn.BatchNorm2d(mlp[0]))
        layers.append(nn.ReLU(inplace=True))

        for i in range(1, len(mlp)):
            layers.append(nn.Conv2d(mlp[i], mlp[i+1], kernel_size=(1, 1)))
            layers.append(nn.BatchNorm2d(mlp[i+1]))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(1,knn_k, stride=1))

        self.netR = nn.Sequential(layers)
    
    def forward(self, x, y):
        pass
