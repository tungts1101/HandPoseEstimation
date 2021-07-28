'''
author: Tran Son Tung
'''

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import open3d as o3d
import scipy.io as sio

subject_names = ["Subject_1", "Subject_2"]
gesture_names = ["put_salt", "use_calculator"]

# ===== load skeleton & object data =====
reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])
cam_extr = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                     [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                     [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                     [0, 0, 0, 1]])
cam_extr_inv = np.linalg.inv(cam_extr)
cam_intr = np.array([[1395.749023, 0, 935.732544],
                     [0, 1395.749268, 540.681030],
                     [0, 0, 1]])

root = '../hand_pose_action'
skeleton_root = os.path.join(root, 'Hand_pose_annotation_v1')
def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    # print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)[sample['frame_idx']]
    return skeleton

pca_root = '../pca'
# ==================================================

class FPHADataset(Dataset):
    def __init__(self, root_path, opt, train=True):
        self.root_path              = root_path
        self.test_index             = opt.test_index
        self.train                  = train
        self.PCA_SZ                 = opt.PCA_SZ
        self.SAMPLE_NUM             = opt.SAMPLE_NUM
        self.JOINT_NUM              = opt.JOINT_NUM
        self.INPUT_FEATURE_NUM      = opt.INPUT_FEATURE_NUM
        self.total_frame_num        = self.__total_frame_num()

        self.point_clouds = np.empty(shape=[self.total_frame_num, self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],
                                     dtype=np.float32)
        # self.volume_length = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)
        self.gt_xyz = np.empty(shape=[self.total_frame_num, self.JOINT_NUM, 3], dtype=np.float32)
        # self.valid = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)

        self.cur_index = 0
        self.start_index = 0
        self.end_index = 0

        if self.train:
            for i_subject in range(len(subject_names)):
                if i_subject != self.test_index:
                    for i_gesture in range(len(gesture_names)):
                        subject = subject_names[i_subject]
                        gesture = gesture_names[i_gesture]
                        self.__load_data(subject, gesture)
        else:
            for i_gesture in range(len(gesture_names)):
                subject = subject_names[self.test_index]
                gesture = gesture_names[i_gesture]
                self.__load_data(subject, gesture)

        self.point_clouds = torch.from_numpy(self.point_clouds)
        # self.volume_length = torch.from_numpy(self.volume_length)
        self.gt_xyz = torch.from_numpy(self.gt_xyz)
        # self.valid = torch.from_numpy(self.valid)

        self.gt_xyz = self.gt_xyz.view(self.total_frame_num, -1)
        # valid_ind = torch.nonzero(self.valid)
        # valid_ind = valid_ind.select(1, 0)

        # self.point_clouds = self.point_clouds.index_select(0, valid_ind.long())
        # self.volume_length = self.volume_length.index_select(0, valid_ind.long())
        # self.gt_xyz = self.gt_xyz.index_select(0, valid_ind.long())
        self.total_frame_num = self.point_clouds.size(0)

        # load PCA coeff
        PCA_coeff_mat = sio.loadmat(os.path.join(pca_root, 'PCA_coeff.mat'))
        self.PCA_coeff = torch.from_numpy(PCA_coeff_mat['PCA_coeff'][:, 0:self.PCA_SZ].astype(np.float32))
        PCA_mean_mat = sio.loadmat(os.path.join(pca_root, 'PCA_mean_xyz.mat'))
        self.PCA_mean = torch.from_numpy(PCA_mean_mat['PCA_mean_xyz'].astype(np.float32))

        tmp = self.PCA_mean.expand(self.total_frame_num, self.JOINT_NUM * 3)
        tmp_demean = self.gt_xyz - tmp
        self.gt_pca = torch.mm(tmp_demean, self.PCA_coeff)

        self.PCA_coeff = self.PCA_coeff.transpose(0, 1).cuda()
        self.PCA_mean = self.PCA_mean.cuda()

    def __len__(self):
        return self.point_clouds.size(0)
    
    def __getitem__(self, index):
        return self.point_clouds[index, :, :], self.gt_pca[index, :], self.gt_xyz[index, :]

    def __total_frame_num(self):
        result = 0

        if self.train:
            for i_subject in range(len(subject_names)):
                if i_subject != self.test_index:
                    for i_gesture in range(len(gesture_names)):
                        subject = subject_names[i_subject]
                        gesture = gesture_names[i_gesture]
                        for _, _, files in os.walk(os.path.join(self.root_path, subject, gesture)):
                            for _ in files:
                                result += 1
        else:
            for i_gesture in range(len(gesture_names)):
                subject = subject_names[self.test_index]
                gesture = gesture_names[i_gesture]
                for _, _, files in os.walk(os.path.join(self.root_path, subject, gesture)):
                    for _ in files:
                        result += 1

        return result

    def __load_data(self, subject, action):
        data_dir = os.path.join(self.root_path, subject, action)

        self.start_index = self.end_index + 1
        for seq_idx in os.listdir(data_dir):
            for path, _, files in os.walk(os.path.join(data_dir, seq_idx)):
                for name in files:
                    pcd = o3d.io.read_point_cloud(os.path.join(path, name))
                    self.point_clouds[self.cur_index,:,:] = np.concatenate((np.asarray(pcd.points).astype(np.float32),np.asarray(pcd.normals).astype(np.float32)), axis=1)

                    # # ===== skeleton & object =====
                    frame_idx = int(name.split('.')[0].split('_')[1])
                    sample = {
                        'subject': subject,
                        'action_name': action,
                        'seq_idx': seq_idx,
                        'frame_idx': frame_idx
                    }
                    skel = get_skeleton(sample, skeleton_root)[reorder_idx]
                    skel *= 1/1000.0 # normalized point with depth scale
                    self.gt_xyz[self.cur_index,:,:] = skel

                    # === Apply camera extrinsic to hand skeleton
                    # skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                    # skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
                    # skel_camcoords = cam_extr_inv.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

                    # skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                    # skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

                    # if seq_idx == '1' and frame_idx == 88 and action == 'use_calculator':
                    #     print(skel)

                    self.cur_index += 1

        self.end_index += self.cur_index
        # self.gt_xyz[(self.start_index - 1):self.end_index, :, :] = gt_data['Volume_GT_XYZ'].astype(np.float32)
        # self.volume_length[(self.start_index - 1):self.end_index, :] = volume_length['Volume_length'].astype(np.float32)
        # self.valid[(self.start_index - 1):self.end_index, :] = valid['valid'].astype(np.float32)
            