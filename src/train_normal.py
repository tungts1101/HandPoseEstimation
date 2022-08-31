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
import utils
from sklearn.model_selection import KFold

from hand_pointnet import PointNet_Plus
from cascaded_pointnet import CascadedNetworkObj
from mydataset import DatasetObj
from mynetwork import NetworkObj
from split_pointnet import SplitPointNet

from wing_loss import WingLoss
import random

import json
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--is_full', '-if', type=bool, default=False)
parser.add_argument('--pca_size', type=int, default=42)
parser.add_argument('--batch_size', '-bs', type=int, default=48)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--save_dir', type=str, default='train_result', help="Folder to save pt file")
parser.add_argument('--model', '-m', type=int, default=1)

parser.add_argument('--ball_radius', '-br', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', '-br2', type=float, default=0.04, help='square of radius for ball query in level 2')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', '-e', type=int, default=50)
parser.add_argument('--weight', '-w', type=str, help="Weight folder")
parser.add_argument('--dataset_folder', '-ds', type=str, default="processed")
parser.add_argument('--contain_obj', '-co', type=bool, default=False)
parser.add_argument('--is_object', '-io', type=int, default=0)
parser.add_argument('--is_normal', '-in', type=int, default=0)
parser.add_argument('--case', '-c', type=str)

parser.add_argument('--device', '-d', type=str, default='cpu')
args = parser.parse_args()

if args.weight:
    save_dir = args.weight
else:
    now = datetime.now()
    now_str = now.strftime("%d-%m-%Y-%H-%M-%S")
    save_dir = os.path.join(args.save_dir, now_str)
    os.makedirs(save_dir, exist_ok=True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=os.path.join(save_dir, 'log.txt'), filemode='a', 
                    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("================================================================================")

if not os.path.exists(os.path.join(save_dir, 'resume.json')):
    state = {
        "epoch": 1,
        "best_err": 1e9
    }
    with open(os.path.join(save_dir, 'resume.json'), "w") as _to:
        json.dump(state, _to, indent=4)

with open(os.path.join(save_dir, 'resume.json'), "r") as _from:
    cur_state = json.load(_from)
    save_state = copy.deepcopy(cur_state)

device = torch.device('cuda:0') if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu'
if not args.weight:
    logging.info("Device: {}".format(device))

### set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

### load data
train_dataset = DatasetObj(is_train=True, is_full=args.is_full, is_obj=(args.is_object == 1), 
                            device=device, dataset_folder=args.dataset_folder)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = DatasetObj(is_train=False, is_full=args.is_full, is_obj=(args.is_object == 1), 
                            device=device, dataset_folder=args.dataset_folder)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if not args.weight:
    logging.info("Train data: {}, Test data: {}".format(len(train_dataset), len(test_dataset)))

### load model
network = None
if args.model == 1:
    network = NetworkObj()
elif args.model == 2:
    network = PointNet_Plus(args.ball_radius2, args.contain_obj)
elif args.model == 3:
    network = CascadedNetworkObj()
elif args.model == 5:
    network = SplitPointNet(args.ball_radius, args.ball_radius2)

network.to(device)
if not args.weight:
    logging.info(network)

criterion = torch.nn.MSELoss().to(device)
# criterion = WingLoss(1, 0.2).to(device)
logging.info("================================================================================\n")


optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas = (0.9, 0.999), eps=1e-05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

if args.weight != None:
    if os.path.exists(os.path.join(args.weight, "network_last.pth")) and \
        os.path.exists(os.path.join(args.weight, "optimizer_last.pth")):
        network.load_state_dict(torch.load(os.path.join(args.weight, "network_last.pth")))
        optimizer.load_state_dict(torch.load(os.path.join(args.weight, "optimizer_last.pth")))
    else:
        if os.path.exists(os.path.join(args.weight, "network_best.pth")) and \
            os.path.exists(os.path.join(args.weight, 'optimizer_best.pth')):
            network.load_state_dict(torch.load(os.path.join(args.weight, 'network_best.pth')))
            optimizer.load_state_dict(torch.load(os.path.join(args.weight, 'optimizer_best.pth')))

best_err = float(cur_state['best_err'])
for epoch in range(int(cur_state['epoch']), args.epoch + 1):
    logging.info("Epoch: {}".format(epoch))
    logging.info("====================")
    
    ## training
    network.train()
    timer = time.time()
    train_mse = 0.0
    train_mse_wld = 0.0

    for i, data in enumerate(tqdm(train_dataloader, 0)):
        points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz, obb_max = data

        estimation = None
        if isinstance(network, NetworkObj):
            estimation = network(points)
        elif isinstance(network, PointNet_Plus):
            inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
            estimation = network(inputs_level1, inputs_level1_center)
        elif isinstance(network, CascadedNetworkObj):
            estimation_stage_1, estimation_stage_2, estimation = network(points, train_dataset.pca_mean, train_dataset.pca_coeff)
        elif isinstance(network, SplitPointNet):
            estimation = network(points)

        loss = None
        if isinstance(network, CascadedNetworkObj):
            loss = 0.25 * criterion(estimation_stage_1, gt_pca) + 0.25 * criterion(estimation_stage_2, gt_pca) + 0.5 * criterion(estimation, gt_pca)
        else:
            obb_len = torch.diff(bound_obb, dim=1)
            if args.contain_obj:
                loss = 0.9 * criterion(estimation[:, :63].reshape(-1, 21, 3) * 63, gt_xyz.reshape(-1, 21, 3) * 63) + \
                    0.1 * criterion(estimation[:, 63:].reshape(-1, 8, 3) * 24, obj_xyz.reshape(-1, 8, 3) * 24)

            else:
                loss = criterion(estimation, gt_xyz)

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_mse = train_mse + loss.data * len(points)
    
    scheduler.step()
    
    logging.info("Time training 1 epoch: {} s".format(time.time() - timer))
    train_mse = train_mse / len(train_dataset)
    logging.info("Train error: {} mm".format(train_mse))

    torch.save(network.state_dict(), os.path.join(save_dir, "network_last.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_last.pth"))

    if best_err > train_mse:
        best_err = float(train_mse)
        logging.info("Save best error: {} mm".format(best_err))
        torch.save(network.state_dict(), os.path.join(save_dir, "network_best.pth"))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_best.pth"))
        save_state["best_err"] = best_err

    save_state["epoch"] = epoch + 1 # for not training this epoch again

    with open(os.path.join(save_dir, 'resume.json'), "w") as _to:
        json.dump(save_state, _to, indent=4)

    if epoch:
        ## testing
        network.eval()
        timer = time.time()
        test_mse = 0.0
        test_wld_err = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_dataloader, 0)):
                points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz, obb_max = data

                ## compute output
                if isinstance(network, NetworkObj):
                    estimation = network(points)
                elif isinstance(network, PointNet_Plus):
                    inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
                    estimation = network(inputs_level1, inputs_level1_center)
                elif isinstance(network, CascadedNetworkObj):
                    estimation_stage_1, estimation_stage_2, estimation = network(points, train_dataset.pca_mean, train_dataset.pca_coeff)
                elif isinstance(network, SplitPointNet):
                    estimation = network(points)

                obb_len = torch.diff(bound_obb, dim=1)
                out_xyz_wld = estimation.data[:, :63].reshape(-1, 21, 3) * obb_len
                gt_xyz_wld = gt_xyz.reshape(-1, 21, 3) * obb_len
                
                diff = torch.pow(out_xyz_wld-gt_xyz_wld, 2).view(-1, 21, 3)
                diff_sum = torch.sum(diff, 2)
                diff_sum_sqrt = torch.sqrt(diff_sum)
                diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
                test_wld_err = test_wld_err + diff_mean.sum()
            
        timer = (time.time() - timer) / len(test_dataset)
        logging.info("Time test 1 sample: {} ms".format(timer * 1000))
        test_wld_err = test_wld_err / len(test_dataset)
        logging.info("Test error in world space: {} mm".format(test_wld_err))