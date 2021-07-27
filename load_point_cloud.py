import open3d as o3d
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fi', required=True, default=0, type=int) # frame index
parser.add_argument('--act', required=True, default='ps', type=str) # frame index
args = parser.parse_args()

action_map = {
    'ps': 'put_salt',
    'uc': 'use_calculator'
}

pcd = o3d.io.read_point_cloud('point_cloud_dataset/Subject_1/{}/1/depth/depth_{:04d}.ply'.format(action_map[args.act], args.fi))
# pcd = o3d.io.read_point_cloud('test_pcd.ply')

# o3d.visualization.draw_geometries([pcd])

print(np.asarray(pcd.points)[:10, :])
print(np.asarray(pcd.normals)[:10, :])