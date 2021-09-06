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
from mydataset import DatasetObj
from mynetwork import NetworkObj
import random

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True, help="Root folder of FPHA dataset")
parser.add_argument('--is_full', type=bool, default=False)
parser.add_argument('--pca_size', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--weight', type=str, required=True, help="Weight folder")
parser.add_argument('--save_dir', type=str, default='eval', help="Folder to save ground truth result")
parser.add_argument('--model', type=int, default=1)

parser.add_argument('--visualize', type=bool, default=False)
parser.add_argument('--subject', type=str, default='')
parser.add_argument('--action', type=str, default='')
parser.add_argument('--seq', type=str, default='')

parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--device', type=str, default='cpu')
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
test_dataset = DatasetObj(root_path=args.root_path, is_train=False, is_full=args.is_full, device=device, subject=args.subject, action=args.action, seq=args.seq)
if args.visualize:
    if args.subject != '' and args.action != '' and args.seq != '':
        args.batch_size = len(glob.glob(os.path.join(args.root_path, 'Video_files', args.subject, args.action, args.seq, 'color', '*.jpeg')))
logging.info("Batch size: {}".format(args.batch_size))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

logging.info("Test data: {}".format(len(test_dataset)))

### load model
### load model
network = None
if args.model == 1:
    network = NetworkObj()
elif args.model == 2:
    network = PointNet_Plus(args.ball_radius2)

network.load_state_dict(torch.load(os.path.join(args.weight, "network_best.pth")))
network.to(device)
# logging.info(network)

criterion = torch.nn.MSELoss(size_average=True).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas = (0.5, 0.999), eps=1e-06)
optimizer.load_state_dict(torch.load(os.path.join(args.weight, "optimizer_best.pth")))
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

network.eval()
with torch.no_grad():
    ## testing
    timer = time.time()
    test_mse = 0.0
    test_mse_wld = 0.0

    for i, data in enumerate(tqdm(test_dataloader, 0)):
        points, gt_pca, gt_xyz, volume_rotate, bound_obb = data

        ## compute output
        estimation = None
        if isinstance(network, NetworkObj):
            estimation = network(points)
        elif isinstance(network, PointNet_Plus):
            inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
            estimation = network(inputs_level1, inputs_level1_center)
        loss = criterion(estimation, gt_pca)

        ## update error
        test_mse = test_mse + loss.item()*len(points)
        pca_mean = test_dataset.pca_mean.expand(estimation.data.size(0), test_dataset.pca_mean.size(1))
        out_xyz = torch.addmm(pca_mean, estimation.data, test_dataset.pca_coeff)

        obb_len = torch.diff(bound_obb, dim=1)
        min_bound = bound_obb[:,:1,:]
        out_xyz_wld = torch.bmm(out_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
        gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)

        if args.visualize:
            if args.subject != '' and args.action != '' and args.seq != '':
                util.visualize(args.root_path, args.subject, args.action, args.seq, test_dataset.valid, out_xyz_wld)

        diff = torch.pow(out_xyz_wld - gt_xyz_wld, 2).view(-1, 21, 3)
        diff_sum_sqrt = torch.sqrt(torch.sum(diff, 2))
        diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
        test_mse_wld = test_mse_wld + diff_mean.sum()
        
    timer = (time.time() - timer) / len(test_dataset)
    logging.info("Time test 1 sample: {} ms".format(timer * 1000))
    test_mse = test_mse / len(test_dataset)
    logging.info("Test MSE 1 sample: {} cm".format(test_mse))
    test_mse_wld = test_mse_wld / len(test_dataset)
    logging.info("Test error 1 sample in world space: {} cm".format(test_mse_wld))

    logging.info("================================================================================\n")