import os
from re import sub
import torch.utils.data
import numpy as np
import glob as glob
import torch
import sys

subject_names_full = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
gesture_names_full = ['charge_cell_phone','clean_glasses','close_juice_bottle','close_liquid_soap','close_milk','close_peanut_butter','drink_mug','flip_pages','flip_sponge', 'give_card',
'give_coin','handshake','high_five','light_candle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet','pour_juice_bottle'
'pour_liquid_soap','pour_milk','pour_wine','prick','put_salt','put_sugar','put_tea_bag','read_letter','receive_coin', 'scoop_spoon','scratch_sponge','sprinkle','squeeze_paper',
'squeeze_sponge','stir','take_letter_from_enveloppe','tear_paper','toast_wine','unfold_glasses','use_calculator','use_flash','wash_sponge','write']
test_subjects_full = ["Subject_2", "Subject_5", "Subject_6"]

subject_names_small = ["Subject_1"]
gesture_names_small = ['put_salt','use_calculator','take_letter_from_enveloppe','open_juice_bottle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet']
test_subjects_small = ["Subject_2"]


class DatasetObj(torch.utils.data.Dataset):
    def __init__(self, root_path, is_train=True, is_full=True, device='cpu',subject='',action='',seq=''):
        self.root_path = root_path
        self.is_train = is_train

        self.start_idx = 0
        self.end_idx = 0

        if is_full:
            self.subject_names = subject_names_full
            self.gesture_names = gesture_names_full
            self.test_subjects = test_subjects_full
        else:
            self.subject_names = subject_names_small
            self.gesture_names = gesture_names_small
            self.test_subjects = test_subjects_small

        if not self.is_train:
            self.subject_names = [subject] if subject != '' else self.subject_names
            self.gesture_names = [action] if action != '' else self.gesture_names
            self.seq = None if seq == '' else seq
        
        self.total_frame_num = self.__total_frame_num()

        self.point_clouds = np.empty(shape=[self.total_frame_num, 1024, 6], dtype=np.float32)
        self.gt_xyz = np.empty(shape=[self.total_frame_num, 63], dtype=np.float32)
        self.valid = np.full(shape=[self.total_frame_num], fill_value=False, dtype=np.bool)
        self.volume_rotate = np.empty(shape=[self.total_frame_num, 3, 3], dtype=np.float32)
        self.bound_obb = np.empty(shape=[self.total_frame_num, 2, 3], dtype=np.float32)

        self.__load_data()

        valid_indexes = np.where(self.valid == True)[0]

        self.point_clouds = torch.from_numpy(np.take(self.point_clouds, valid_indexes, axis=0)).to(device)
        self.gt_xyz = np.take(self.gt_xyz, valid_indexes, axis=0)
        self.volume_rotate = torch.from_numpy(np.take(self.volume_rotate, valid_indexes, axis=0)).to(device)
        self.bound_obb = torch.from_numpy(np.take(self.bound_obb, valid_indexes, axis=0)).to(device)

        self.total_frame_num = self.point_clouds.size(0)

        # calculate pca
        pca_coeff_mat = np.load('../pca_mean/PCA_coeff.npy').astype(np.float32)
        pca_mean = np.load('../pca_mean/PCA_mean.npy').astype(np.float32)

        tmp = np.broadcast_to(pca_mean, (self.total_frame_num, 63))
        tmp_diff = (self.gt_xyz - tmp)
        self.gt_pca = torch.from_numpy(np.dot(tmp_diff, np.transpose(pca_coeff_mat))).to(device)

        self.gt_xyz = torch.from_numpy(self.gt_xyz).to(device)
        self.pca_mean = torch.from_numpy(np.expand_dims(pca_mean, axis=0)).to(device)
        self.pca_coeff = torch.from_numpy(pca_coeff_mat).to(device)

    def __getitem__(self, index):
        return self.point_clouds[index, :, :], self.gt_pca[index, :], self.gt_xyz[index, :], self.volume_rotate[index, :, :], self.bound_obb[index, :, :] 

    def __len__(self):
        return self.point_clouds.size(0)
    
    def __load_data(self):
        if self.is_train:
            for i_subject in self.subject_names:
                if i_subject in self.test_subjects: continue
                for i_gesture in self.gesture_names:
                    gesture_folder = os.path.join('../processed', i_subject, i_gesture)
                    if not os.path.exists(gesture_folder): continue
                    for seq_idx in os.listdir(gesture_folder):
                        self.__load_data_dir(os.path.join(gesture_folder, seq_idx))
        else:
            for i_subject in self.test_subjects:
                for i_gesture in self.gesture_names:
                    gesture_folder = os.path.join('../processed', i_subject, i_gesture)
                    if not os.path.exists(gesture_folder): continue
                    for seq_idx in os.listdir(gesture_folder):
                        if self.seq != None and seq_idx != self.seq: continue
                        self.__load_data_dir(os.path.join(gesture_folder, seq_idx))

    def __load_data_dir(self, seq_folder):
        point_cloud = np.load(os.path.join(seq_folder, 'points.npy')).astype(np.float32)
        gt_xyz = np.load(os.path.join(seq_folder, 'gt_xyz.npy')).astype(np.float32)
        bound_obb = np.load(os.path.join(seq_folder, 'bound_obb.npy')).astype(np.float32)
        volume_rotate = np.load(os.path.join(seq_folder, 'volume_rotate.npy')).astype(np.float32)
        valid = np.load(os.path.join(seq_folder, 'valid.npy')).astype(np.bool)

        self.start_idx = self.end_idx + 1
        self.end_idx = self.end_idx + point_cloud.shape[0]

        self.point_clouds[(self.start_idx-1):self.end_idx, :, :] = point_cloud
        self.gt_xyz[(self.start_idx-1):self.end_idx, :] = gt_xyz
        self.bound_obb[(self.start_idx-1):self.end_idx, :, :] = bound_obb
        self.volume_rotate[(self.start_idx-1):self.end_idx, :, :] = volume_rotate
        self.valid[(self.start_idx-1):self.end_idx] = valid

    def __total_frame_num(self):
        total = 0
        if self.is_train:
            for i_subject in self.subject_names:
                if i_subject in self.test_subjects: continue
                for i_gesture in self.gesture_names:
                    if sys.platform.startswith('win32'):
                        total += len(glob.glob(os.path.join(self.root_path, 'Video_files', i_subject, i_gesture, '**\*.jpeg'), recursive=True))
                    else:
                        total += len(glob.glob(os.path.join(self.root_path, 'Video_files', i_subject, i_gesture, '**/*.jpeg'), recursive=True))
        else:
            for i_subject in self.test_subjects:
                for i_gesture in self.gesture_names:
                    if self.seq != None:
                        total += len(glob.glob(os.path.join(self.root_path, 'Video_files', i_subject, i_gesture, self.seq, 'color', '*.jpeg'), recursive=True))
                    else:
                        if sys.platform.startswith('win32'):
                            total += len(glob.glob(os.path.join(self.root_path, 'Video_files', i_subject, i_gesture, '**\*.jpeg'), recursive=True))
                        else:
                            total += len(glob.glob(os.path.join(self.root_path, 'Video_files', i_subject, i_gesture, '**/*.jpeg'), recursive=True))
        
        return total