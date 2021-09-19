import random
import numpy as np
import open3d as o3d
from mylib import knn_search, random_walk

class HandModel:
    def __init__(self, xyz, method="rw"):
        self.__build(xyz, method)
    
    def __build(self, xyz, method):
        N, C = xyz.shape
        idx = [random.randrange(0,N) for _ in range(21)]
        centers = np.take(xyz, idx, axis=0)
        # print(centers)
        # self.__show(centers)

        points = []

        if method == "knn": # generate with knn
            around_indices = np.take(knn_search(xyz, k=32), idx, axis=0)
            for i, i_center in enumerate(centers):
                points.append(i_center)
                around_points = np.take(xyz, around_indices[i], axis=0)
                for point in around_points:
                    points.append(point)
        elif method == "rw": # generate with random walk
            for i_center in centers:
                points.append(i_center)
                around_points = random_walk(i_center, 64, 0.1)
                for point in around_points:
                    points.append(point)
        
        # self.__show(points)
        self.__show(xyz)

    def __show(self, points):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=10))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

        radii = [0.05, 0.01, 0.02, 0.04]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))
        o3d.visualization.draw_geometries([rec_mesh])

        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=11)

        # o3d.visualization.draw_geometries([mesh], zoom=1,
        #                                 front=[-0.4761, -0.4698, -0.7434],
        #                                 lookat=[0, 0, 0],
        #                                 up=[0.2304, -0.8825, 0.4101])
        # # o3d.visualization.draw_geometries([pcd])