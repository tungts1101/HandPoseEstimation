import os
import numpy as np
import shutil
import cv2
import argparse
import random

subject_names = ["Subject_1", "Subject_3", "Subject_4"]
action_names = ["put_salt"]

def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    # print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    if len(skeleton_vals) == 0: return np.array([])
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)
    return skeleton

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help="Path to dataset install")
    parser.add_argument('--save_dir', required=True, help="Path to save images and labels")
    parser.add_argument('--total', required=True, type=int, help="Total files to generate")

    args = parser.parse_args()
    img_root = os.path.join(args.root, "Video_files")
    skeleton_root = os.path.join(args.root, 'Hand_pose_annotation_v1')

    reorder_idx = np.array([
        0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
        20
    ])
    cam_extr = np.array(
        [[0.999988496304, -0.00468848412856, 0.000982563360594,
          25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
         [-0.000969709653873, 0.00274303671904, 0.99999576807,
          3.902], [0, 0, 0, 1]])
    cam_intr = np.array([[1395.749023, 0, 935.732544],
                         [0, 1395.749268, 540.681030], [0, 0, 1]])

    count = 0
    for path, subdirs, _ in os.walk(img_root):
        for dir in subdirs:
            if dir != "color": continue
            path = path.replace('\\', '/')
            subject, action_name, seq_idx = path[len(args.root)+1:].split('/')[1:4]
            if int(seq_idx) != 1: continue

            if subject not in subject_names: continue
            if action_name not in action_names: continue # comment this line to process in all actions
            sample = {
                "subject": subject,
                "action_name": action_name,
                "seq_idx": seq_idx
            }
            skeleton = get_skeleton(sample, skeleton_root)
            if len(skeleton) == 0: continue

            for _, _, files in os.walk(os.path.join(path, dir)):
                for name in files:
                    filename, fileextension = os.path.splitext(name)
                    
                    frame_idx = filename.split("_")[1].lstrip("0")
                    frame_idx = "0" if frame_idx == '' else frame_idx

                    # copy images file
                    filepath = os.path.join(path, dir, name)
                    savename = '_'.join([subject, action_name, seq_idx, frame_idx, name])
                    savepath = os.path.join(args.save_dir, "images", savename)
                    print(savepath)
                    shutil.copy(filepath, savepath)

                    # generate bounding box in yolo format
                    skel = skeleton[int(frame_idx)][reorder_idx]
                    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
                    skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                    skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]

                    imagemat = cv2.imread(filepath)
                    image_height, image_width, _ = imagemat.shape

                    # print(image_width, image_height)
                    # print(skel_proj)

                    for skel_val in skel_proj:
                        imagemat = cv2.circle(imagemat, (int(skel_val[0]), int(skel_val[1])), 3, (255, 0, 0), -1)

                    x_min = max(0, min(skel_proj[:,0]) - 100)
                    x_max = min(image_width, max(skel_proj[:,0]) + 100)
                    y_min = max(0, min(skel_proj[:,1]) - 100)
                    y_max = min(image_height, max(skel_proj[:,1]) + 100)

                    # print(x_min, x_max, y_min, y_max)

                    x_center_norm = ((x_max + x_min) / 2) / image_width
                    y_center_norm = ((y_max + y_min) / 2) / image_height
                    x_width_norm = (x_max - x_min) / image_width
                    y_width_norm = (y_max - y_min) / image_height

                    # print(x_center_norm, y_center_norm, x_width_norm, y_width_norm)
                    # print(x_width_norm * image_width)

                    # # visualize bounding box
                    # center = (int(image_width * x_center_norm), int(image_height * y_center_norm))
                    # start_point = (center[0] - int(image_width * x_width_norm / 2), center[1] - int(image_height * y_width_norm / 2))
                    # end_point = (center[0] + int(image_width * x_width_norm / 2), center[1] + int(image_height * y_width_norm / 2))
                    # color = (255, 0, 0)
                    # thickness = 2
                    # imagemat = cv2.rectangle(imagemat, start_point, end_point, color, thickness)
                    # cv2.imshow("BoundingBox", imagemat)
                    # cv2.waitKey(0)

                    save_label_name = savename[:-len(fileextension)] + ".txt"
                    save_label_path = os.path.join(args.save_dir, "labels", save_label_name)

                    # print(save_label_path)

                    with open(save_label_path, "w") as f:
                        f.write('0 {} {} {} {}'.format(x_center_norm, y_center_norm, x_width_norm, y_width_norm))
                        f.close()

                    count += 1
                    if count >= args.total:
                        print("Generating done")
                        exit()
