import numpy as np
import os
import cv2

def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    # print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, 3)
    return skeleton

def visualize_joints_2d(image, gt_joints, es_joints):
    """Draw 2d skeleton on matplotlib axis"""
    links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]

    _draw2djoints(image, gt_joints, links, col=(0, 0, 0))
    _draw2djoints(image, es_joints, links, None)

def _draw2djoints(image, joints, links, col):
    """Draw segments, one color per link"""
    colors = [
        (255, 0, 0),
        (0, 255, 255),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255)
    ]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            pt1 = [int(x) for x in joints[finger_links[idx]]]
            pt2 = [int(x) for x in joints[finger_links[idx+1]]]
            cv2.line(image, pt1, pt2, color=col if col is not None else colors[finger_idx], thickness=5)
        
        for idx in range(len(finger_links)):
            pt = [int(x) for x in joints[finger_links[idx]]]
            cv2.circle(image, pt, 5, color=col if col is not None else colors[finger_idx], thickness=-1)

def visualize(root_path, subject, action, seq, valid_idx, estimated_xyz):
    reorder_idx = np.array([
        0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
        20
    ])

    sample = {
        "subject": subject,
        "action_name": action,
        "seq_idx": seq
    }

    cam_extr = np.array(
        [[0.999988496304, -0.00468848412856, 0.000982563360594,
          25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
         [-0.000969709653873, 0.00274303671904, 0.99999576807,
          3.902], [0, 0, 0, 1]])
    cam_intr = np.array([[1395.749023, 0, 935.732544],
                         [0, 1395.749268, 540.681030], [0, 0, 1]])
    
    def get_skel(skel):
        # Apply camera extrinsic to hand skeleton
        skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        skel_camcoords = cam_extr.dot(
            skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

        skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
        skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

        return skel_proj

    skeleton_root = os.path.join(root_path, 'Hand_pose_annotation_v1')
    gt_skels = get_skeleton(sample, skeleton_root)[:, reorder_idx]

    i_image = 0
    i_valid = 0
    folder = os.path.join(root_path, 'Video_files', subject, action, seq, 'color')
    for i_image, image_file in enumerate(os.listdir(folder)):
        if not valid_idx[i_image]:
            continue
        
        gt_skel = gt_skels[i_valid]
        es_skel = estimated_xyz[i_valid]

        img = cv2.imread(os.path.join(folder, image_file))

        gt_skel_proj = get_skel(gt_skel)
        es_skel_proj = get_skel(es_skel)
        visualize_joints_2d(img, gt_skel_proj, es_skel_proj)

        cv2.imshow('img', img)
        cv2.waitKey(0)

        i_valid += 1