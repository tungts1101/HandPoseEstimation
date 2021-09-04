from numpy.lib.arraysetops import unique
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

class HandModel:
    def __init__(self, pcd):
        # inititalize hand model from a point cloud dataset
        self._pcd = None
        self.construct_point_clouds(pcd)

    def show(self):
        o3d.visualization.draw_geometries(self._pcd)
    
    def construct_point_clouds(self, pcd):
        pcd_points = np.asarray(pcd.points)
        tree = KDTree(pcd_points)

        obb = pcd.get_oriented_bounding_box()
        center = obb.get_center()
        max_bound = obb.get_max_bound()
        min_bound = obb.get_min_bound()

        radius = ((max_bound - min_bound) / 2) * 0.9 # multiply with 0.9 as bounding is larger than the actual hand
        radius = (radius * radius).sum(axis=0) ** 0.5

        # palm
        palm_center = center - radius * np.array([0.25, 0.25, 0.25])
        palm_distance = radius * 0.2
        palm_num_points = 256
        palm_point_cloud = self.generate_point_clouds(pcd_points, tree, palm_center, palm_distance, palm_num_points)

        joint_dirs = [
            np.array([-1, 0, 0]), #WRIST

            np.array([-0.4, 0.5, 0]), #TMCP
            np.array([0, 0.3, 0]), #IMCP
            np.array([0, 0.0, 0]), #MMCP
            np.array([0, -0.3, 0]), #RMCP
            np.array([0, -0.5, 0]), #PMCP

            np.array([-0.2, 0.5, 0]), #TPIP
            np.array([0.0, 0.5, 0]), #TDIP
            np.array([0.2, 0.5, 0]), #TTIP

            np.array([0.2, 0.3, 0]), #IPIP
            np.array([0.4, 0.3, 0]), #IDIP
            np.array([0.6, 0.3, 0]), #ITIP

            np.array([0.2, 0.0, 0]), #MPIP
            np.array([0.4, 0.0, 0]), #MDIP
            np.array([0.7, 0.0, 0]), #MTIP

            np.array([0.2, -0.3, 0]), #RPIP
            np.array([0.4, -0.3, 0]), #RDIP
            np.array([0.6, -0.3, 0]), #RTIP

            np.array([0.2, -0.5, 0]), #PPIP
            np.array([0.3, -0.5, 0]), #PDIP
            np.array([0.5, -0.5, 0]), #PTIP
        ]
        # [Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP]
        joints = list(map(lambda x: self.generate_point_clouds_for_joint(pcd_points, tree, center, radius, x), joint_dirs))
        joints.append(palm_point_cloud)

        self._pcd = joints
        # o3d.visualization.draw_geometries(self._pcd)
    
    def generate_point_clouds_for_joint(self, pcd_points, tree, center, radius, direction):
        joint_center = center + radius * direction
        joint_distance = radius * 0.08
        joint_num_points = 32
        return self.generate_point_clouds(pcd_points, tree, joint_center, joint_distance, joint_num_points)

    def generate_point_clouds(self, pcd_points, tree, center, distance, num_points):
        dd, ii = tree.query(center, k=num_points, distance_upper_bound=distance)
        ii = unique(ii)

        points = np.array([pcd_points[i-1] for i in ii])
        point = center + (distance * 0.1 * np.random.random_sample((3,)) - distance * 0.1)

        while len(points) < num_points:
            point += (distance * 0.1 * np.random.random_sample((3,)) - distance * 0.1)

            dis = np.sum((point - center)**2)

            if dis > distance ** 2:
                point = center
            
            points = np.vstack((points, point))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))