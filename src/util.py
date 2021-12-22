import numpy as np
import os
import cv2
from scipy.spatial import KDTree
from PIL import Image
import matplotlib.pyplot as plt

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

def visualize_joints(gt_joints, es_joints):
    links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    
    minimum = np.minimum(gt_joints, es_joints)
    maximum = np.maximum(gt_joints, es_joints)

    _min = np.amin(minimum, axis=0)
    _max = np.amax(maximum, axis=0)
    _max -= _min

    _max /= 3

    if _max[0] > 1000 or _max[1] > 1000: return []

    image = np.full([int(_max[1]+1), int(_max[0]+1), 3],fill_value=255,dtype=np.uint8)

    gt_joints -= _min
    es_joints -= _min

    gt_joints /= 3
    es_joints /= 3

    _draw2djoints(image, gt_joints, links, col=(0,0,0))
    _draw2djoints(image, es_joints, links, None)

    return image

def _draw2djoints(image, joints, links, col):
    """Draw segments, one color per link"""
    colors = [
        (0, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 0, 255),
    ]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            pt1 = [int(x) for x in joints[finger_links[idx]]]
            pt2 = [int(x) for x in joints[finger_links[idx+1]]]
            cv2.line(image, pt1, pt2, color=col if col is not None else colors[finger_idx], thickness=2)
        
        for idx in range(len(finger_links)):
            pt = [int(x) for x in joints[finger_links[idx]]]
            cv2.circle(image, pt, 3, color=col if col is not None else colors[finger_idx], thickness=-1)

def visualize(gt, es):
    reorder_idx = np.array([
        0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
        20
    ])

    links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    cols = ['m', 'b', 'g', 'y', 'r']

    for i in range(gt.shape[0]):
        i_gt = np.squeeze(gt[i, :].reshape(-1, 21, 3)[:, reorder_idx].cpu())
        i_es = np.squeeze(es[i, :].reshape(-1, 21, 3)[:, reorder_idx].cpu())

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xs = i_gt[:, 0]
        ys = i_gt[:, 1]
        zs = i_gt[:, 2]
        ax.scatter(xs, ys, zs, marker='o')

        for link in links:
            ax.plot([xs[i] for i in link], [ys[i] for i in link], [zs[i] for i in link], color='k')

        xs = i_es[:, 0]
        ys = i_es[:, 1]
        zs = i_es[:, 2]
        ax.scatter(xs, ys, zs, marker='8')

        for idx, link in enumerate(links):
            ax.plot([xs[i] for i in link], [ys[i] for i in link], [zs[i] for i in link], color=cols[idx])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()