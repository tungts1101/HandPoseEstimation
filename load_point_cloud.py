import open3d as o3d
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fi', required=True, default=0, type=int) # frame index
args = parser.parse_args()
pcd = o3d.io.read_point_cloud('point_cloud_dataset/Subject_1/put_salt/1/depth/depth_{:04d}.ply'.format(args.fi))
# print(pcd)
# print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])