from tqdm import tqdm
from multiprocessing import Pool
import cv2
import numpy as np
import open3d as o3d
import os
from sklearn.cluster import DBSCAN
from collections import Counter
import glob

from tqdm.utils import IS_NIX, IS_WIN

base_root = 'hand_pose_action\\Video_files' if IS_WIN else 'hand_pose_action/Video_files'
subjects = ['Subject_1', 'Subject_2']
pcd_root = 'point_cloud_dataset'

num_point_sampling = 1024
depth_threshold = 550

def generate_point_cloud(filepath):
    pcd_filepath = os.path.join(pcd_root, filepath[len(base_root)+1:].split('.')[-2] + '.ply')

    if not os.path.exists(pcd_filepath):
        open(pcd_filepath, 'w').close()
    
    # ===== read image =====
    depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    # ===== crop all parts too far from camera =====
    depth = cv2.split(depth_img)[0]
    depth[depth>depth_threshold] = 0

    # ===== convert numpy array to open3d image =====
    image = o3d.geometry.Image(depth.astype(np.uint16))
    width, height = np.asarray(image).shape
    intrinsic_mat = o3d.camera.PinholeCameraIntrinsic(width, height, 475.065948, 475.065857, 315.944855, 245.287079)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(image, intrinsic_mat, depth_scale=1000.0)
    ## flip the pointcloud
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # ===== downsample by voxel =====
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd_points = np.asarray(pcd.points)
    if len(pcd_points) == 0:
        with open("warning_file.txt", "a") as warning_file:
            warning_file.write(pcd_filepath)
            return

    # ===== clustering point cloud =====
    clustering = DBSCAN(eps=0.008,min_samples=10,algorithm='kd_tree').fit(pcd_points)
    labels = clustering.labels_

    if len(labels) == 0:
        return

    counter = Counter(labels)
    cluster_distance = {}

    for point_idx, point in enumerate(pcd.points):
        label = labels[point_idx]
        if counter[label] < 100: continue
        
        if not label in cluster_distance:
            cluster_distance[label] = 0
        
        distance = np.sum(point**2)
        cluster_distance[label] += distance

    for label in cluster_distance:
        cluster_distance[label] /= counter[label]

    hand_label = min(cluster_distance, key=cluster_distance.get)
    pcd_points = np.array([point for (i, point) in enumerate(pcd_points) if labels[i] == hand_label])

    # ===== downsample with farthest point sampling =====
    def cal_dis(p0, points):
        return ((p0 - points)**2).sum(axis=1)

    def farthest_point_sampling(pts, k):
        if len(pts) < k:
            return [i for i in range(len(pts))] + [np.random.randint(len(pts)) for _ in range(k - len(pts))]

        indices = np.zeros((k, ), dtype=np.uint32)
        indices[0] = np.random.randint(len(pts))
        min_distances = cal_dis(pts[indices[0]], pts)
        for i in range(1, k):
            indices[i] = np.argmax(min_distances)
            min_distances = np.minimum(min_distances, cal_dis(pts[indices[i]], pts))
        return indices

    indices = farthest_point_sampling(pcd_points, num_point_sampling)
    pcd.points = o3d.utility.Vector3dVector([pcd_points[idx] for idx in indices])
    pcd.transform(np.linalg.inv(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])))

    # estimate and orient normal to camera location
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000.0) # multiply with scale to get real world position

    o3d.io.write_point_cloud(pcd_filepath, pcd)

def worker(subject):
    path = os.path.join(base_root, subject) + '\\**\\*.png' if IS_WIN else os.path.join(base_root, subject) + '/**/*.png'
    files = glob.glob(path, recursive=True)

    subfolder = os.path.join(pcd_root, subject)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    for filepath in tqdm(files, position=subjects.index(subject)):
        generate_point_cloud(filepath)

if __name__ == '__main__':
    with Pool(4) as p:
        p.map(worker, subjects)
        p.close()
        p.join()