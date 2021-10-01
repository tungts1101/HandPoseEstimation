import argparse
import os
import trimesh
import numpy as np
from mydataset import obj_contained_action, subject_names_full

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
args = parser.parse_args()

obj_trans_root = os.path.join(args.root, 'Object_6D_pose_annotation_v1_1')
obj_root = os.path.join(args.root, 'Object_models')

obj_map_with_action = {
    'close_juice_bottle': 'juice_bottle',
    'close_liquid_soap': 'liquid_soap',
    'close_milk': 'milk', 
    'open_juice_bottle': 'juice_bottle', 
    'open_liquid_soap': 'liquid_soap', 
    'open_milk': 'milk', 
    'pour_juice_bottle': 'juice_bottle', 
    'pour_liquid_soap': 'liquid_soap', 
    'pour_milk': 'milk', 
    'put_salt': 'salt'
}

def load_objects(obj_root):
    object_names = ['juice_bottle', 'liquid_soap', 'milk', 'salt']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = mesh
    return all_models

def get_obj_transform(sample, obj_root):
    seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                            sample['seq_idx'], 'object_pose.txt')
    with open(seq_path, 'r') as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample['frame_idx']]
    line = raw_line.strip().split(' ')
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    # print('Loading obj transform from {}'.format(seq_path))
    return trans_matrix

object_infos = load_objects(obj_root)

for i_subject in subject_names_full:
    for i_gesture in obj_contained_action:
        gesture_folder = os.path.join('..\processed', i_subject, i_gesture).replace('\\', '/')
        if not os.path.exists(gesture_folder): continue
        for seq_idx in os.listdir(gesture_folder):
            if not seq_idx.isnumeric(): continue

            bound_obb = np.load(os.path.join(gesture_folder, seq_idx, 'bound_obb.npy'))
            volume_rotate = np.load(os.path.join(gesture_folder, seq_idx, 'volume_rotate.npy'))
            valid_frame_idx = np.load(os.path.join(gesture_folder, seq_idx, 'valid.npy')).astype(np.bool)

            # i_gt_xyz = np.matmul(i_gt_xyz, obb.R.transpose())
            # i_gt_xyz = (i_gt_xyz - min_bound) / obb_len
            num_frame = bound_obb.shape[0]
            points = np.zeros((num_frame, 8, 3)).astype(np.float32)

            for frame_idx in range(num_frame):
                if not valid_frame_idx[frame_idx]: continue
                sample = {
                    'subject': i_subject,
                    'action_name': i_gesture,
                    'seq_idx': seq_idx,
                    'frame_idx': frame_idx
                }

                obj_trans = get_obj_transform(sample, obj_trans_root)
                mesh = object_infos[obj_map_with_action[i_gesture]]
                verts = np.array(mesh.bounding_box_oriented.vertices) * 1000

                hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
                verts_trans = obj_trans.dot(hom_verts.T).T
                verts_trans = verts_trans[:, :-1]

                verts_trans = np.matmul(verts_trans, volume_rotate[frame_idx].transpose())
                verts_trans = (verts_trans - bound_obb[frame_idx][0]) / (bound_obb[frame_idx][1] - bound_obb[frame_idx][0])
                points[frame_idx, :, :] = verts_trans
        
            np.save(os.path.join(gesture_folder, seq_idx, 'obj_xyz.npy'), points)