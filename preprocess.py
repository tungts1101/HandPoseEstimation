from multiprocessing import Pool
import cv2
import numpy as np
import open3d as o3d
import os
from sklearn.cluster import DBSCAN
from collections import Counter
import sys

root = 'hand_pose_action/Video_files'
pcd_root = 'point_cloud_dataset'
cur_file_counter = 0
num_point_sampling = 1024
depth_threshold = 550

def generate_point_cloud_one_image(path, name):
    global cur_file_counter

    subject_path = path[len(root)+1:]
    subject_path = subject_path.replace('\\', '/')  # for compatibility between windows and linux

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

    # ===== clustering point cloud =====
    clustering = DBSCAN(eps=0.008,min_samples=10,algorithm='kd_tree').fit(pcd_points)
    labels = clustering.labels_

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

    cur_file_counter += 1
    progress(count=cur_file_counter, total=total, status='Preprocessing depth image')

def generate_point_cloud():
    for path, subdirs, files in os.walk(root):
        for name in files:
            generate_point_cloud_one_image(path, name)

# shout out to https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)

total = 0
def calculate_total_files():
    global total

    for path, subdirs, files in os.walk(root):
        for name in files:
            subject_path = path[len(root)+1:]
            subject_path = subject_path.replace('\\', '/')  # for compatibility between windows and linux

            if subject_path.split('/')[-1] == 'depth':
                total += 1

calculate_total_files()
generate_point_cloud()