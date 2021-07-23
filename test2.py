import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import random as rd
from collections import Counter
from sklearn.cluster import DBSCAN

filepath = 'hand_pose_action/Video_files/Subject_1/use_calculator/1/depth/depth_0088.png'
# filepath = 'hand_pose_action/Video_files/Subject_1/put_salt/1/depth/depth_0025.png'
depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

# ===== crop all parts too far from camera =====
depth = cv2.split(depth_img)[0]
depth[depth>550] = 0

# depth_checked_hand_obj = depth / 256.0
# cv2.imshow('Depth checked', depth_checked_hand_obj)
# cv2.waitKey(0)

# ===== convert numpy array to open3d image =====
image = o3d.geometry.Image(depth.astype(np.uint16))
width, height = np.asarray(image).shape
intrinsic_mat = o3d.camera.PinholeCameraIntrinsic(width, height, 475.065948, 475.065857, 315.944855, 245.287079)
extrinsic_mat = np.array([
                    [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7], 
                    [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                    [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902], 
                    [0, 0, 0, 1]
                ])
pcd = o3d.geometry.PointCloud.create_from_depth_image(image, intrinsic_mat, extrinsic_mat)
## flip the pointcloud
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
## uncomment to view the point cloud
# o3d.visualization.draw_geometries([pcd])

pcd = pcd.voxel_down_sample(voxel_size=0.005)
pcd_points = np.asarray(pcd.points)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    # labels = np.array(pcd.cluster_dbscan(eps=0.015, min_points=10, print_progress=True))
    clustering = DBSCAN(eps=0.008,min_samples=10,algorithm='kd_tree').fit(pcd_points)
    labels = clustering.labels_

# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd])

counter = Counter(labels)
cluster_distance = {}

for point_idx, point in enumerate(pcd.points):
    label = labels[point_idx]
    if counter[label] < 100: continue
    
    if not label in cluster_distance:
        cluster_distance[label] = 0
    
    distance = (-point[0] - 25.77) ** 2 + (point[1] - 1.22) ** 2 + (point[2] - 3.902) ** 2
    cluster_distance[label] += distance

for label in cluster_distance:
    cluster_distance[label] /= counter[label]

# print(cluster_distance)

hand_label = min(cluster_distance, key=cluster_distance.get)

pcd.points = o3d.utility.Vector3dVector([point for (i, point) in enumerate(pcd_points) if labels[i] == hand_label])
pcd_points = np.asarray(pcd.points)
print(pcd_points.shape)
# o3d.visualization.draw_geometries([pcd])

# farthest point sampling
def cal_dis(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def farthest_point_sampling(pts, k):
    if len(pts) < k:
        return [i for i in range(pts)] + [rd.randint(len(pts)) for _ in range(k - len(pts))]

    indices = np.zeros((k, ), dtype=np.uint32)
    indices[0] = np.random.randint(len(pts))
    min_distances = cal_dis(pts[indices[0]], pts)
    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        min_distances = np.minimum(min_distances, cal_dis(pts[indices[i]], pts))
    return indices

indices = farthest_point_sampling(pcd_points, 512)

pcd.points = o3d.utility.Vector3dVector([pcd_points[i] for i in indices])
print(pcd.points)
o3d.visualization.draw_geometries([pcd])