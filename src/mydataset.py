import os
from re import sub
import torch.utils.data
import numpy as np
import glob as glob
import torch
import sys

subject_names_full = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
gesture_names_full = ['charge_cell_phone','clean_glasses','close_juice_bottle','close_liquid_soap','close_milk','close_peanut_butter','drink_mug','flip_pages','flip_sponge', 'give_card',
'give_coin','handshake','high_five','light_candle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet','pour_juice_bottle',
'pour_liquid_soap','pour_milk','pour_wine','prick','put_salt','put_sugar','put_tea_bag','read_letter','receive_coin', 'scoop_spoon','scratch_sponge','sprinkle','squeeze_paper',
'squeeze_sponge','stir','take_letter_from_enveloppe','tear_paper','toast_wine','unfold_glasses','use_calculator','use_flash','wash_sponge','write']

subject_names_small = ["Subject_1", "Subject_2"]
gesture_names_small = ['put_salt','use_calculator','take_letter_from_enveloppe','open_juice_bottle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet']

obj_contained_action = ['close_juice_bottle', 'close_liquid_soap', 'close_milk', 'open_juice_bottle', 'open_liquid_soap', 
'open_milk', 'pour_juice_bottle', 'pour_liquid_soap', 'pour_milk', 'put_salt']
test_gestures = ["put_salt"]

train_subject = ["Subject_1", "Subject_2", "Subject_4", "Subject_5", "Subject_6"]
test_subject  = ["Subject_3"]

class DatasetObj(torch.utils.data.Dataset):
    def __init__(self, is_train=True, is_full=True, device='cpu', is_obj=False, subject=None,action=None, seq=None, dataset_folder='processed'):
        self.is_train = is_train
        self.is_obj = is_obj
        self.__dataset_folder = dataset_folder

        self.start_idx = 0
        self.end_idx = 0
        self.seq = None

        if is_full:
            self.subject_names = subject_names_full
            self.gesture_names = gesture_names_full
            if self.is_obj:
                self.gesture_names = obj_contained_action
        else:
            self.subject_names = subject_names_small
            self.gesture_names = gesture_names_small
            if self.is_obj:
                self.gesture_names = obj_contained_action

        if self.is_train:
            self.subject_names = train_subject
        else:
            self.subject_names = test_subject

        if not self.is_train:
            self.subject_names = [subject] if subject else self.subject_names
            self.gesture_names = [action] if action else self.gesture_names
            self.seq = seq if seq else None
        
        # if self.is_train:
        #     self.subject_names = train_subject
        #     self.gesture_names = gesture_names_full
        # else:
        #     self.subject_names = test_subject
        #     self.gesture_names = gesture_names_full

        self.total_frame_num = self.__total_frame_num()
        print("Subjects: {}\nGestures: {}\nTotal frame num: {}\n".format(self.subject_names, self.gesture_names, self.total_frame_num))

        self.point_clouds = np.empty(shape=[self.total_frame_num, 1024, 6], dtype=np.float32)
        self.gt_xyz = np.empty(shape=[self.total_frame_num, 63], dtype=np.float32)
        self.valid = np.full(shape=[self.total_frame_num], fill_value=False, dtype=np.bool)
        self.volume_rotate = np.empty(shape=[self.total_frame_num, 3, 3], dtype=np.float32)
        self.bound_obb = np.empty(shape=[self.total_frame_num, 2, 3], dtype=np.float32)
        self.obj_xyz = np.empty(shape=[self.total_frame_num, 8, 3], dtype=np.float32)

        self.__load_data()

        valid_indexes = np.where(self.valid == True)[0]

        self.point_clouds = torch.from_numpy(np.take(self.point_clouds, valid_indexes, axis=0)).to(device)
        self.gt_xyz = np.take(self.gt_xyz, valid_indexes, axis=0)
        self.volume_rotate = torch.from_numpy(np.take(self.volume_rotate, valid_indexes, axis=0)).to(device)
        self.bound_obb = torch.from_numpy(np.take(self.bound_obb, valid_indexes, axis=0)).to(device)
        self.obj_xyz = torch.from_numpy(np.take(self.obj_xyz, valid_indexes, axis=0)).to(device)

        self.total_frame_num = self.point_clouds.size(0)

        if self.total_frame_num == 0:
            return

        # calculate pca
        pca_coeff_mat = np.load('../pca_mean/PCA_coeff.npy').astype(np.float32)
        pca_mean = np.load('../pca_mean/PCA_mean.npy').astype(np.float32)

        tmp = np.broadcast_to(pca_mean, (self.total_frame_num, 63))
        tmp_diff = (self.gt_xyz - tmp)
        self.gt_pca = torch.from_numpy(np.dot(tmp_diff, np.transpose(pca_coeff_mat))).to(device)

        self.gt_xyz = torch.from_numpy(self.gt_xyz).to(device)
        self.pca_mean = torch.from_numpy(np.expand_dims(pca_mean, axis=0)).to(device)
        self.pca_coeff = torch.from_numpy(pca_coeff_mat).to(device)

        obb_len = torch.diff(self.bound_obb, dim=1)
        self.obb_max = torch.max(obb_len, dim=0).values.to(device)

    def __getitem__(self, index):
        return self.point_clouds[index, :, :], self.gt_pca[index, :], \
            self.gt_xyz[index, :], self.volume_rotate[index, :, :], \
            self.bound_obb[index, :, :], self.obj_xyz[index, :, :], \
            self.obb_max 

    def __len__(self):
        return self.total_frame_num
    
    def __load_data(self):
        for i_subject in self.subject_names:
            for i_gesture in self.gesture_names:
                gesture_folder = os.path.join('..', self.__dataset_folder, i_subject, i_gesture)
                if not os.path.exists(gesture_folder): continue
                for i_seq in os.listdir(gesture_folder):
                    try:
                        if not i_seq.isnumeric(): continue
                        if self.is_obj and not os.path.exists(os.path.join(gesture_folder, i_seq, 'obj_xyz.npy')): continue
                        if self.seq and i_seq != self.seq: continue
                        self.__load_data_dir(os.path.join(gesture_folder, i_seq))
                    except Exception as e:
                        print(e)

    def __load_data_dir(self, seq_folder):
        point_cloud = np.load(os.path.join(seq_folder, 'points.npy')).astype(np.float32)
        gt_xyz = np.load(os.path.join(seq_folder, 'gt_xyz.npy')).astype(np.float32)
        bound_obb = np.load(os.path.join(seq_folder, 'bound_obb.npy')).astype(np.float32)
        volume_rotate = np.load(os.path.join(seq_folder, 'volume_rotate.npy')).astype(np.float32)
        valid = np.load(os.path.join(seq_folder, 'valid.npy')).astype(np.bool)

        if self.is_obj:
            obj_xyz = np.load(os.path.join(seq_folder, 'obj_xyz.npy')).astype(np.float32)

        self.start_idx = self.end_idx + 1
        self.end_idx = self.end_idx + point_cloud.shape[0]

        self.point_clouds[(self.start_idx-1):self.end_idx, :, :] = point_cloud
        self.gt_xyz[(self.start_idx-1):self.end_idx, :] = gt_xyz
        self.bound_obb[(self.start_idx-1):self.end_idx, :, :] = bound_obb
        self.volume_rotate[(self.start_idx-1):self.end_idx, :, :] = volume_rotate
        self.valid[(self.start_idx-1):self.end_idx] = valid

        if self.is_obj:
            self.obj_xyz[(self.start_idx-1):self.end_idx] = obj_xyz

    def __total_frame_num(self):
        total = 0

        for i_subject in self.subject_names:
            for i_gesture in self.gesture_names:
                try:
                    gesture_folder = os.path.join('..', self.__dataset_folder, i_subject, i_gesture)
                    if not os.path.exists(gesture_folder): continue
                    for i_seq in os.listdir(gesture_folder):
                        try:
                            if self.seq and i_seq != self.seq: continue
                            seq_valid_path = os.path.join(gesture_folder, i_seq, 'valid.npy')
                            if not os.path.exists(seq_valid_path): continue
                            valid = np.load(seq_valid_path).astype(np.bool)
                            total += valid.shape[0]
                        except Exception as e:
                            print(e)
                except Exception as e:
                    print(e)
        return total