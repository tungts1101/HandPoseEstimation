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
    d = np.load('../processed/Subject_1/put_salt/1/valid.npy').astype(np.bool)
    e = np.load('../processed/Subject_1/put_salt/1/volume_rotate.npy')


    xyz_norm = b[0]

    pca_coeff = np.load('../pca_mean/PCA_coeff.npy')
    pca_mean = np.load('../pca_mean/PCA_mean.npy')

    # print(pca_mean)

    tmp = np.broadcast_to(pca_mean, xyz_norm.shape)
    gt_pca = np.dot(xyz_norm - tmp, np.transpose(pca_coeff))

    # print(gt_pca)
    print(pca_coeff)

    xyz = np.dot(gt_pca, pca_coeff) + tmp
    obb_len = a[0][1] - a[0][0]
    gt_xyz = xyz.reshape(-1, 21, 3) * obb_len + a[0][0]
    gt_xyz = np.matmul(gt_xyz, e[0]).flatten()

    print(gt_xyz)





