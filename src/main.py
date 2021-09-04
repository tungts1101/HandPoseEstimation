import os
import open3d as o3d
from hand_model import HandModel
import scipy.io as sio
import numpy as np

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

    a = np.load('../processed/Subject_1/put_salt/1/bound_obb.npy')
    b = np.load('../processed/Subject_1/put_salt/1/gt_xyz.npy')
    c = np.load('../processed/Subject_1/put_salt/1/points.npy')
    d = np.load('../processed/Subject_1/put_salt/1/valid.npy')
    e = np.load('../processed/Subject_1/put_salt/1/volume_rotate.npy')

    obb_len = a[0][1] - a[0][0]
    print(obb_len.shape)
    gt_xyz = b[0].reshape(-1, 21, 3) * obb_len + a[0][0]
    gt_xyz = np.matmul(gt_xyz, e[0]).flatten()
    print(gt_xyz.shape)

    # PCA_coeff_mat = sio.loadmat('../pca/PCA_coeff.mat')
    # PCA_coeff = np.array(PCA_coeff_mat['PCA_coeff'][:, 0:42].astype(np.float32))

    # PCA_mean_mat = sio.loadmat('../pca/PCA_mean_xyz.mat')
    # PCA_mean = np.array(PCA_mean_mat['PCA_mean_xyz'].astype(np.float32))

    # # print(PCA_coeff_mat)
    # print(PCA_coeff.shape)
    # # print(PCA_coeff)

    # # print(PCA_mean)
    # print(PCA_mean.shape)

    pca_coeff = np.load('../pca_mean/PCA_coeff.npy')
    pca_mean = np.load('../pca_mean/PCA_mean.npy')

    pca_coeff = pca_coeff[:, :42]
    tmp = np.broadcast_to(pca_mean, gt_xyz.shape)
    gt_pca = np.matmul(gt_xyz - tmp, pca_coeff)

    print(gt_pca.shape)
    print(gt_pca)

