import cv2
import numpy as np
import open3d as o3d
import os
import trimesh
import time
import argparse
from collections import Counter
from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser()
parser.add_argument('--override', '-o', type=bool, default=False)
parser.add_argument('--source', '-s', type=str, default="/media/data3/datasets/F-PHAB")
parser.add_argument('--save_dir', '-sd', type=str, default="no_hand_detect_model_processed")
args = parser.parse_args()

subject_names = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
# subject_names = ["Subject_1"]
# gesture_names = ['charge_cell_phone','clean_glasses','close_juice_bottle','close_liquid_soap','close_milk','close_peanut_butter','drink_mug','flip_pages','flip_sponge', 'give_card',
# 'give_coin','handshake','high_five','light_candle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet','pour_juice_bottle'
# 'pour_liquid_soap','pour_milk','pour_wine','prick','put_salt','put_sugar','put_tea_bag','read_letter','receive_coin', 'scoop_spoon','scratch_sponge','sprinkle','squeeze_paper',
# 'squeeze_sponge','stir','take_letter_from_enveloppe','tear_paper','toast_wine','unfold_glasses','use_calculator','use_flash','wash_sponge','write']
#gesture_names = ['put_salt','use_calculator','take_letter_from_enveloppe','open_juice_bottle','open_juice_bottle','open_letter','open_liquid_soap','open_milk','open_peanut_butter','open_soda_can','open_wallet']
gesture_names = ['close_juice_bottle', 'close_liquid_soap', 'close_milk', 'open_juice_bottle', 'open_liquid_soap', 
'open_milk', 'pour_juice_bottle', 'pour_liquid_soap', 'pour_milk', 'put_salt']
# gesture_names = ['put_salt']

def generate_point_cloud_from_depth(depth_val,is_visualize=False, voxel_size=3.0, dbscan_eps=8.0,use_voxel_downsample=True):
    img_height, img_width = depth_val.shape

    # ===== crop all parts too far from camera =====
    if is_visualize:
        vis_depth = depth_val / 256.0
        cv2.imshow("Depth image", vis_depth)
        cv2.waitKey(0)

    # ===== convert numpy array to open3d image =====
    image = o3d.geometry.Image(depth_val.astype(np.uint16))
    intrinsic_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, 475.065948, 475.065857, 315.944855, 245.287079)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(image, intrinsic_mat, depth_scale=1.0, project_valid_depth_only=True)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    if use_voxel_downsample:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # if is_visualize:
    #     o3d.visualization.draw_geometries([pcd])

    pcd_points = np.asarray(pcd.points)
    if len(pcd_points) == 0:
        print("Not enough point")
        return (None, None)

    pcd_points *= np.array([-1, -1 , 1])

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=10, print_progress=True))

    labels = DBSCAN(eps=dbscan_eps,min_samples=10,algorithm='kd_tree').fit(pcd_points).labels_

    if len(labels) == 0:
        print("Cannot cluster point cloud")
        return (None, None)

    # if is_visualize:
    #     max_label = labels.max()
    #     print(f"point cloud has {max_label + 1} clusters")
    #     colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #     colors[labels < 0] = 0
    #     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #     o3d.visualization.draw_geometries([pcd])

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

    if len(cluster_distance) != 0:
        hand_label = min(cluster_distance, key=cluster_distance.get)

        pcd.points = o3d.utility.Vector3dVector([point for (i, point) in enumerate(pcd_points) if labels[i] == hand_label])
        pcd_points = np.asarray(pcd.points)

        # if is_visualize:
        #     o3d.visualization.draw_geometries([pcd])

    # farthest point sampling
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

    # if len(pcd_points) < 1024:
    #     obb = pcd.get_oriented_bounding_box()
    #     center = obb.get_center()
    #     max_bound = obb.get_max_bound()
    #     min_bound = obb.get_min_bound()
    #     radius = ((max_bound - min_bound) / 2.0) * 0.5

    #     rd_points = 2 * radius * np.random.random_sample((1024 - len(pcd_points), 3)) + (center - radius)
    #     pcd_points = np.vstack((pcd_points, rd_points))
    #     pcd.points = o3d.utility.Vector3dVector(pcd_points)

    #     # o3d.visualization.draw_geometries([pcd])
    # else:
    indices = farthest_point_sampling(pcd_points, 1024)
    pcd.points = o3d.utility.Vector3dVector([pcd_points[i] for i in indices])

    # pcd.transform(np.linalg.inv(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])))

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=dbscan_eps, max_nn=10))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    # o3d.visualization.draw_geometries([pcd])
    # print(np.asarray(pcd.points)[:10, :])
    # print("======")
    # print(np.asarray(pcd.normals)[:10, :])
    # print("######")

    # normalize by obb
    # obb = pcd.get_oriented_bounding_box()
    # rotate_mat_transpose = obb.R.transpose()
    # pcd.rotate(rotate_mat_transpose, obb.get_center())

    # if is_visualize:
    #     o3d.visualization.draw_geometries([pcd])

    # print(np.asarray(pcd.points)[:10, :])
    # print("======")
    # print(np.asarray(pcd.normals)[:10, :])
    # print("######")

    # norm = np.linalg.norm(pcd.points)
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points / norm))

    if is_visualize:
        o3d.visualization.draw_geometries([pcd])

    # return (pcd, pcd.get_oriented_bounding_box())
    return (pcd, o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.array(pcd.points))))

def generate_point_cloud():
    source = args.source
    save_dir = args.save_dir

    video_dir = os.path.join(source, 'Video_files')
    hand_annotation = os.path.join(source, 'Hand_pose_annotation_v1')

    cam_extr = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                     [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                     [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                     [0, 0, 0, 1]])
    
    obj_trans_root = os.path.join(source, 'Object_6D_pose_annotation_v1_1')
    obj_root = os.path.join(source, 'Object_models')

    obj_map_with_action = {
        'close_juice_bottle': 'juice_bottle',
        'close_liquid_soap': 'liquid_soap',
        'close_milk': 'milk', 
        'open_juice_bottle': 'juice_bottle', 
        'open_liquid_soap': 'liquid_soap', 
        'open_milk': 'milk', 
        'pour_juice_bottle': 'juice_bottle', 
        'pour_liquid_soap': 'liquid_soap', 
        'pour_milk': 'milk', 
        'put_salt': 'salt'
    }

    def load_objects(obj_root):
        object_names = ['juice_bottle', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            all_models[obj_name] = mesh
        return all_models

    def get_obj_transform(sample, obj_root):
        seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        # print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    object_infos = load_objects(obj_root)

    def cam_coords(world_coord):
        if world_coord.ndim == 1:
            world_coord_mat = np.concatenate([world_coord, np.ones(1)])
            return cam_extr.dot(world_coord_mat.transpose()).transpose()[:3].astype(np.float32)
        else:
            B, S = world_coord.shape
            if S == 3:
                world_coord_mat = np.concatenate([world_coord, np.ones([B, 1])], 1)

            return cam_extr.dot(world_coord_mat.transpose()).transpose()[:, :3].astype(np.float32)

    for subject in subject_names:
        print("Subject: {}".format(subject))
        time1 = time.time()
        for gesture in gesture_names:
            try:
                if not os.path.exists(os.path.join(video_dir, subject, gesture)): continue
                print("Gesture: {}".format(gesture))
                for seq_idx in os.listdir(os.path.join(video_dir, subject, gesture)):
                    if not seq_idx.isnumeric(): continue
                    if not os.path.exists(os.path.join(video_dir, subject, gesture, seq_idx)): continue
                    # save files
                    save_seq_path = os.path.join(save_dir, subject, gesture, seq_idx)
                    if not args.override and os.path.exists(save_seq_path): continue
                    filesize = os.path.getsize(os.path.join(hand_annotation, subject, gesture, seq_idx, 'skeleton.txt'))
                    if filesize == 0: continue
    
                    if not os.path.exists(save_seq_path):
                        os.makedirs(save_seq_path)
                    
                    print("Seq idx: {}".format(seq_idx))
    
                    # read ground truth joint
                    gt_ws = np.loadtxt(os.path.join(hand_annotation, subject, gesture, seq_idx, 'skeleton.txt')).astype(np.float32)
                    gt_ws = gt_ws[:,1:]
    
                    frame_num = gt_ws.shape[0]
                    
                    points = np.zeros((frame_num, 1024, 6)).astype(np.float32) # xyz + norm for each point
                    volume_rotate = np.zeros((frame_num, 3, 3)).astype(np.float32) # rotation matrix
                    bound_obb = np.zeros((frame_num, 2, 3)).astype(np.float32) # min bound & max bound of rotation matrix
                    gt_xyz = np.zeros((frame_num, 63)).astype(np.float32)
                    obj_xyz = np.zeros((frame_num, 8, 3)).astype(np.float32) # coordinate of obj
                    valid = [False for _ in range(frame_num)]
    
                    image_dir = os.path.join(video_dir, subject, gesture, seq_idx, 'depth')

                    for dirpath, _, files in os.walk(image_dir):
                        for image in files:
                            filepath = os.path.join(dirpath, image)
                            depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

                            # ===== crop all parts too far from camera =====
                            depth = cv2.split(depth_img)[0]
                            depth[depth>550] = 0

                            # depth_checked_hand_obj = depth / 256.0
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
                            # x,y,w,h = cv2.boundingRect(c)

                            # ===== show cropped hand and obj =====
                            mask = np.zeros(depth_img.shape[0:2], dtype=np.uint8)
                            cv2.drawContours(mask, [c], -1, color=(255, 255, 255), thickness=cv2.FILLED) # Draw filled contour in mask
                            # cv2.imshow('Mask', mask)
                            # cv2.waitKey(0)

                            crop_depth = np.zeros_like(depth_img)
                            crop_depth[mask == 255] = depth_img[mask == 255]
        
                            pcd, obb = generate_point_cloud_from_depth(crop_depth)

                            if pcd != None:
                                filepath = filepath.replace('\\', '/')
                                file_name = filepath.split('/')[-1][:-4]
                                frame_idx = int(file_name.split('.')[0].split('_')[1])

                                valid[frame_idx] = True
                                volume_rotate[frame_idx] = obb.R

                                pcd.rotate(obb.R.transpose(), obb.get_center())
                                pcd_points = np.asarray(pcd.points)
                                min_bound = np.min(np.array(pcd_points), axis=0)
                                max_bound = np.max(np.array(pcd_points), axis=0)
                                bound_obb[frame_idx] = np.asarray([min_bound, max_bound])
                                obb_len = max_bound - min_bound + 1e-6

                                # min_bound_camcoords = cam_coords(min_bound)
                                # max_bound_camcoords = cam_coords(max_bound)
                                # bound_obb[frame_idx] = np.asarray([min_bound_camcoords, max_bound_camcoords])
                                # obb_len = max_bound_camcoords - min_bound_camcoords + 1e-6

                                pcd_normals = np.asarray(pcd.normals)
                                # pcd_points_camcoords = cam_coords(pcd_points)

                                # pcd.points = pcd_points_camcoords
                                # pcd.normals = pcd_normal_camcoords
                                # pcd_camcoords = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points))
                                # pcd_camcoords.normals = o3d.utility.Vector3dVector(pcd_normals)
                                # o3d.visualization.draw_geometries([pcd_camcoords])

                                pcd_points = (pcd_points - min_bound) / obb_len
                                # pcd_points_camcoords = (pcd_points_camcoords - min_bound_camcoords) / obb_len
                                # print(np.all(pcd_points <= 1.0))
                                # print(np.all(pcd_points >= -1.0))

                                points[frame_idx] = np.concatenate((pcd_points, pcd_normals), axis=1)

                                # rotate & normalize ground truth
                                i_gt_xyz = gt_ws[frame_idx].reshape((21, 3))
                                i_gt_xyz = np.matmul(i_gt_xyz, obb.R.transpose())
                                i_gt_xyz = (i_gt_xyz - min_bound) / obb_len
                                gt_xyz[frame_idx] = i_gt_xyz.flatten()
                                # i_gt_xyz_camcoords = cam_coords(i_gt_xyz)
                                # i_gt_xyz_camcoords = (i_gt_xyz_camcoords - min_bound_camcoords) / obb_len
                                # gt_xyz[frame_idx] = i_gt_xyz_camcoords.flatten()

                                # calculate obj pose
                                sample = {
                                    'subject': subject,
                                    'action_name': gesture,
                                    'seq_idx': seq_idx,
                                    'frame_idx': frame_idx
                                }

                                obj_trans = get_obj_transform(sample, obj_trans_root)
                                mesh = object_infos[obj_map_with_action[gesture]]
                                verts = np.array(mesh.bounding_box_oriented.vertices) * 1000

                                hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
                                verts_trans = obj_trans.dot(hom_verts.T).T
                                verts_trans = verts_trans[:, :-1]

                                # rotate & normalize
                                verts_trans = np.matmul(verts_trans, obb.R.transpose())
                                verts_trans = (verts_trans - min_bound) / obb_len
                                obj_xyz[frame_idx] = verts_trans

                    np.save(os.path.join(save_seq_path, 'points.npy'), points)
                    np.save(os.path.join(save_seq_path, 'volume_rotate.npy'), volume_rotate)
                    np.save(os.path.join(save_seq_path, 'bound_obb.npy'), bound_obb)
                    np.save(os.path.join(save_seq_path, 'gt_xyz.npy'), gt_xyz)
                    np.save(os.path.join(save_seq_path, 'valid.npy'), valid)
                    np.save(os.path.join(save_seq_path, 'obj_xyz.npy'), obj_xyz)
            except Exception as e:
                print(e)

    print('Done for {} in {}s'.format(subject, time.time() - time1))

generate_point_cloud()