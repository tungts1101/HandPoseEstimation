import os
from mylib import random_walk_torch
import open3d as o3d
from hand_model_v2 import HandModel
import scipy.io as sio
import numpy as np
import torch
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import farthest_point_sample

if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud('../point_cloud_dataset_norm_v3/Subject_1/put_salt/1/depth_0015.ply')
    # model = HandModel(pcd)
    # model.show()

    # PCA_mean_mat = sio.loadmat('../pca/PCA_mean_xyz.mat')
    # PCA_mean = np.array(PCA_mean_mat['PCA_mean_xyz'].astype(np.float32))
    # PCA_coeff_mat = sio.loadmat('../pca/PCA_coeff.mat')
    # PCA_coeff = np.array(PCA_coeff_mat['PCA_coeff'][:, 0:42].astype(np.float32))

    # print(PCA_coeff.shape)
    # print(PCA_mean.shape)

    # pcd = o3d.io.read_point_cloud('../point_cloud_dataset_norm_v3/Subject_1/use_calculator/1/depth_0001.ply')
    # xyz = np.asarray(pcd.points)
    # norm = np.asarray(pcd.normals).astype(np.float32)

    # print(xyz.shape)
    # print(norm.shape)

    # xyz = np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals)))
    
    # print(xyz.shape)

    # a = np.load('../processed/Subject_1/put_salt/1/bound_obb.npy')
    # b = np.load('../processed/Subject_1/put_salt/1/gt_xyz.npy')
    # c = np.load('../processed/Subject_1/put_salt/1/points.npy')
    # d = np.load('../processed/Subject_1/put_salt/1/valid.npy').astype(np.bool)
    # e = np.load('../processed/Subject_1/put_salt/1/volume_rotate.npy')


    # xyz_norm = b[0]

    # pca_coeff = np.load('../pca_mean/PCA_coeff.npy')
    # pca_mean = np.load('../pca_mean/PCA_mean.npy')

    # # # print(pca_mean)

    # tmp = np.broadcast_to(pca_mean, xyz_norm.shape)
    # gt_pca = np.dot(xyz_norm - tmp, np.transpose(pca_coeff))

    # # # print(gt_pca)
    # # print(pca_coeff)

    # xyz = np.dot(gt_pca, pca_coeff) + tmp
    # obb_len = a[0][1] - a[0][0]
    # gt_xyz = xyz.reshape(-1, 21, 3) * obb_len + a[0][0]
    # gt_xyz = np.matmul(gt_xyz, e[0]).flatten()

    # print(gt_xyz)

    # frame_idx = 20

    # xyz = c[:,:,:3]
    # xyz = xyz * obb_len + a[0][0]
    # xyz = np.matmul(xyz, e[0])

    # xyz = xyz[frame_idx] # get frame idx
    # print(xyz.shape)

    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    # # o3d.visualization.draw_geometries([pcd])
    # model = HandModel(pcd)
    # model.show()

    # xyz = xyz[frame_idx]
    # model = HandModel(xyz)
    device = 'cpu'

    # arr = torch.rand((64, 1024, 3)).to(device)
    # idx = farthest_point_sample(arr, 64)
    # print(idx[0])
    # print(idx.shape)

    arr = torch.rand((64, 32, 16)).to(device)
    print(arr[0][0])
    idx = random_walk_torch(arr, 16, 0.05)
    print(idx[0][:5,:])