from imp import NullImporter
from pyexpat import model
from scipy import rand
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
from mydataset import DatasetObj, gesture_names_full
from mynetwork import NetworkObj
import random

import scipy.io as sio

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

parser.add_argument('--ball_radius', '-br', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', '-br2', type=float, default=0.04, help='square of radius for ball query in level 2')
parser.add_argument('--case', '-c', type=str)
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

### load model
network = None
if args.model == 2:
    network = PointNet_Plus(args.ball_radius2, args.contain_obj)
elif args.model == 5:
    network = SplitPointNet(args.ball_radius, args.ball_radius2)

network.load_state_dict(torch.load(os.path.join(args.weight, "network_best.pth")))
network.to(device)
# logging.info(network)

criterion = torch.nn.MSELoss(size_average=True).to(device)

network.eval()

### load data
dataset = DatasetObj(is_train=False, is_full=args.is_full, is_obj=(args.is_object == 1), 
                    device=device, dataset_folder=args.dataset_folder,
                    subject=args.subject, action=args.action, seq=args.seq)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
logging.info("Dataset: {}".format(len(dataset)))

with torch.no_grad():
    ## testing
    timer = time.time()
    # test_mse = 0.0
    test_wld_err = 0.0
    error_per_frame = np.zeros(len(dataset)).astype(np.float32)
    max_err_per_frame = np.zeros(len(dataset)).astype(np.float32)
    err_per_joints = np.zeros((len(dataset), 21)).astype(np.float32)
    last_frame_index = 0

    for i, data in enumerate(tqdm(dataloader, 0)):
        points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz, obb_max = data

        ## compute output
        if isinstance(network, PointNet_Plus):
            inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
            estimation = network(inputs_level1, inputs_level1_center)
        elif isinstance(network, SplitPointNet):
            estimation = network(points)

        obb_len = torch.diff(bound_obb, dim=1)

        out_xyz_wld = estimation.data[:, :63].reshape(-1, 21, 3) * obb_len
        gt_xyz_wld = gt_xyz.reshape(-1, 21, 3) * obb_len

        if args.visualize:
            if args.subject != '' and args.action != '' and args.seq != '':
                util.visualize(gt_xyz_wld, out_xyz_wld)
        
        diff = torch.pow(out_xyz_wld-gt_xyz_wld, 2).view(-1, 21, 3)
        diff_sum = torch.sum(diff, 2)
        diff_sum_sqrt = torch.sqrt(diff_sum)
        diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
        test_wld_err = test_wld_err + diff_mean.sum()

        num_frame = points.shape[0]
        error_per_frame[last_frame_index:last_frame_index+num_frame] = np.squeeze(diff_mean.cpu())
        # max_err_per_frame[last_frame_index:last_frame_index+num_frame] = np.amax(diff_sum_sqrt.cpu().numpy(), axis=1)
        # err_per_joints[last_frame_index:last_frame_index+num_frame] = diff_sum_sqrt.cpu()
        last_frame_index = last_frame_index + num_frame
    
    model_name = ""
    if args.model == 2:
        model_name = "hpn_whd"
    elif args.model == 5:
        model_name = "spn"

    # err_per_joints = np.sum(err_per_joints, axis=0) / len(dataset)
    # np.save(os.path.join("eval", "summerize", "epj_{}.npy".format(model_name)), err_per_joints)
    # np.save(os.path.join("eval", "summerize", "mepf_{}.npy".format(model_name)), max_err_per_frame)
    np.save(os.path.join("eval", "summerize", 'epf_{}_case{}.npy'.format(model_name, args.case)), error_per_frame)
    timer = (time.time() - timer) / len(dataset)
    # logging.info("Time test 1 sample: {} ms".format(timer * 1000))
    # test_mse = test_mse / len(test_dataset)
    # logging.info("Test MSE 1 sample: {} cm".format(test_mse))
    test_wld_err = test_wld_err / len(dataset)
    logging.info("Test error 1 sample in world space: {} mm".format(test_wld_err))
    logging.info("================================================================================\n")