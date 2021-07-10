import cv2
import numpy as np
import open3d
import os

root = 'hand_pose_action/Video_files'
pcd_root = 'point_cloud_dataset'

def generate_point_cloud_one_image(path, name):
    subject_path = path[len(root)+1:]
    if subject_path.split('/')[-1] != 'depth': return   # consider only depth image

    filename, _ = os.path.splitext(name)
    pcd_filepath = os.path.join(pcd_root, subject_path, filename + '.ply')
    subfolder = os.path.join(pcd_root, subject_path)

    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    if not os.path.exists(pcd_filepath):
        open(pcd_filepath, 'w').close()

    # ===== read image =====
    filepath = os.path.join(path, name)
    depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # ===== crop all parts too far from camera =====
    depth = cv2.split(depth_img)[0]
    depth[depth>550] = 0

    depth_checked_hand_obj = depth / 256.0
    # cv2.imshow('Depth checked', depth_checked_hand_obj)
    # cv2.waitKey(0)

    # ===== find the largest contour =====
    depth_u8 = depth.astype('uint8')
    threshold_value = int(np.max(depth_u8) * 0.1)
    _, threshold = cv2.threshold(depth_u8, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1,1),np.uint8)
    dilate = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, 3)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    # ===== show cropped hand and obj =====
    mask = np.zeros(depth_img.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED) # Draw filled contour in mask
    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)

    out = np.zeros_like(depth_img)
    out[mask == 255] = depth_img[mask == 255]

    ## uncomment to view the mask of hand and object cropped by depth based and contour based
    # contour_checked_hand_obj = out / 256.0
    # numpy_horizontal = np.hstack((depth_checked_hand_obj, contour_checked_hand_obj))
    # cv2.imshow('Cropped Hand Object', numpy_horizontal)
    # cv2.waitKey(0)

    ## region of interest
    # roi = out[y:y+h,x:x+w]

    # ===== convert numpy array to open3d image =====
    image = open3d.geometry.Image(out.astype(np.uint16))
    width, height = np.asarray(image).shape
    intrinsic_mat = open3d.camera.PinholeCameraIntrinsic(width, height, 475.065948, 475.065857, 315.944855, 245.287079)
    extrinsic_mat = np.array([
                        [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7], 
                        [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                        [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902], 
                        [0, 0, 0, 1]
                    ])
    pcd = open3d.geometry.PointCloud.create_from_depth_image(image, intrinsic_mat, extrinsic_mat)
    ## flip the pointcloud
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    ## uncomment to view the point cloud
    # open3d.visualization.draw_geometries([pcd])

    open3d.io.write_point_cloud(pcd_filepath, pcd)

def generate_point_cloud():
    for path, subdirs, files in os.walk(root):
        for name in files:
            generate_point_cloud_one_image(path, name)

generate_point_cloud()