import torch
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import os
from datetime import datetime
import logging
from tqdm import tqdm
import sys
import time
import torch.utils.data
import numpy as np
import glob

import util
import utils

from hand_pointnet import PointNet_Plus
from split_pointnet import SplitPointNet
from mydataset import DatasetObj
from mynetwork import NetworkObj
import random

parser = argparse.ArgumentParser()
parser.add_argument('--is_full', '-if', type=bool, default=False)
parser.add_argument('--pca_size', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--save_dir', type=str, default='eval', help="Folder to save ground truth result")
parser.add_argument('--model', '-m', type=int, default=1)

parser.add_argument('--visualize', '-v', type=bool, default=False)
parser.add_argument('--subject', '-s', type=str)
parser.add_argument('--action', '-a', type=str)
parser.add_argument('--seq', type=str)

parser.add_argument('--weight', '-w', type=str, help="Weight folder")
parser.add_argument('--dataset_folder', '-ds', type=str, default="processed")
parser.add_argument('--contain_obj', '-co', type=bool, default=False)
parser.add_argument('--is_object', '-io', type=int, default=0)
parser.add_argument('--is_normal', '-in', type=int, default=0)

parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

parser.add_argument('--device', '-d', type=str, default='cpu')
args = parser.parse_args()

now = datetime.now()
now_str = now.strftime("%d-%m-%Y-%H-%M-%S")
save_dir = os.path.join(args.save_dir, now_str)
os.makedirs(save_dir)

logging.basicConfig(filename=os.path.join(save_dir, 'log.txt'), filemode='w', 
                    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("================================================================================")

device = torch.device('cuda:0') if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu'
logging.info("Device: {}".format(device))

### set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

### load data
dataset = DatasetObj(is_train=False, is_full=args.is_full, is_obj=(args.is_object == 1), 
                    device=device, dataset_folder=args.dataset_folder, is_normal=(args.is_normal == 1),
                    subject=args.subject, action=args.action, seq=args.seq)
# if args.visualize:
#     if args.subject != '' and args.action != '' and args.seq != '':
#         args.batch_size = len(glob.glob(os.path.join(args.root_path, 'Video_files', args.subject, args.action, args.seq, 'color', '*.jpeg')))
# logging.info("Batch size: {}".format(args.batch_size))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

logging.info("Dataset: {}".format(len(dataset)))

### load model
### load model
network = None
if args.model == 1:
    network = NetworkObj()
elif args.model == 2:
    network = PointNet_Plus(args.ball_radius2, args.contain_obj)
elif args.model == 5:
    network = SplitPointNet(args.ball_radius, args.ball_radius2)

network.load_state_dict(torch.load(os.path.join(args.weight, "network_best.pth")))
network.to(device)
# logging.info(network)

criterion = torch.nn.MSELoss(size_average=True).to(device)

network.eval()
with torch.no_grad():
    ## testing
    timer = time.time()
    # test_mse = 0.0
    test_wld_err = 0.0

    for i, data in enumerate(tqdm(dataloader, 0)):
        points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz, obb_max = data

        obb_len = torch.diff(bound_obb, dim=1)
        points[:, :, :3] = points[:, :, :3] * obb_len / obb_max
        # gt_xyz = gt_xyz.reshape(-1, 21, 3) * obb_len / obb_max
        # gt_xyz = gt_xyz.reshape(-1, 63)

        ## compute output
        if isinstance(network, NetworkObj):
            estimation = network(points)
        elif isinstance(network, PointNet_Plus):
            inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
            estimation = network(inputs_level1, inputs_level1_center)
        elif isinstance(network, SplitPointNet):
            estimation = network(points)

        # eval_loss = None
        # if isinstance(network, CascadedNetworkObj):
        #     eval_loss = 0.25 * criterion(estimation_stage_1, gt_pca) + 0.25 * criterion(estimation_stage_2, gt_pca) + 0.5 * criterion(estimation, gt_pca)
        # else:
        #     obb_len = torch.diff(bound_obb, dim=1)
        #     if args.contain_obj:
        #         # eval_loss = criterion(estimation, torch.cat((gt_xyz, obj_xyz.reshape(-1, 24)), dim=1)) * 1000
        #         # eval_loss = 0.9 * criterion(estimation[:, :63].reshape(-1, 21, 3) * obb_len, gt_xyz.reshape(-1, 21, 3) * obb_len) + \
        #         #     0.1 * criterion(estimation[:, 63:].reshape(-1, 8, 3) * obb_len, obj_xyz.reshape(-1, 8, 3) * obb_len)

        #         # eval_loss = 0.8 * criterion(estimation[:, :63].reshape(-1, 21, 3) * 100, gt_xyz.reshape(-1, 21, 3) * 100) + \
        #         #     0.2 * criterion(estimation[:, 63:].reshape(-1, 8, 3) * 100, obj_xyz.reshape(-1, 8, 3) * 100)
        #         eval_loss = (0.9 * criterion(estimation[:, :63].reshape(-1, 21, 3), gt_xyz.reshape(-1, 21, 3)) + \
        #             0.1 * criterion(estimation[:, 63:].reshape(-1, 8, 3), obj_xyz.reshape(-1, 8, 3))) * 1000
        #     else:
        #         eval_loss = criterion(estimation, gt_xyz) * 1000
        #         # eval_loss = criterion(estimation * 100, gt_xyz * 100)
        #         # eval_loss = criterion(estimation.reshape(-1, 21, 3) * obb_len, gt_xyz.reshape(-1, 21, 3) * obb_len)

        ## update error
        # test_mse = test_mse + eval_loss.item()

        # obb_len = torch.diff(bound_obb, dim=1)
        # min_bound = bound_obb[:,:1,:]
        # out_xyz_wld = torch.bmm(estimation.data[:, :63].reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
        # gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
        # out_xyz_wld = estimation.data[:, :63].reshape(-1, 21, 3) * obb_len
        # gt_xyz_wld = gt_xyz.reshape(-1, 21, 3) * obb_len

        out_xyz_wld = estimation.data[:, :63].reshape(-1, 21, 3) * obb_max
        gt_xyz_wld = gt_xyz.reshape(-1, 21, 3) * obb_len

        if args.visualize:
            if args.subject != '' and args.action != '' and args.seq != '':
                util.visualize(gt_xyz_wld, out_xyz_wld)
        
        diff = torch.pow(out_xyz_wld-gt_xyz_wld, 2).view(-1, 21, 3)
        diff_sum = torch.sum(diff, 2)
        diff_sum_sqrt = torch.sqrt(diff_sum)
        diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
        test_wld_err = test_wld_err + diff_mean.sum()
        # diff_mean_wld = torch.mul(diff_mean, obb_len)
        # test_wld_err = test_wld_err + diff_mean_wld.sum()

        # eval_loss = criterion(estimation, gt_xyz)
        # test_wld_err = test_wld_err + eval_loss.item()
        
    # timer = (time.time() - timer) / len(test_dataset)
    # logging.info("Time test 1 sample: {} ms".format(timer * 1000))
    # test_mse = test_mse / len(test_dataset)
    # logging.info("Test MSE 1 sample: {} cm".format(test_mse))
    test_wld_err = test_wld_err / len(dataset)
    logging.info("Test error 1 sample in world space: {} cm".format(test_wld_err))

    logging.info("================================================================================\n")