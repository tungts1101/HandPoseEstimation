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

from hand_pointnet import PointNet_Plus
from cascaded_pointnet import CascadedNetworkObj
from mydataset import DatasetObj
from mynetwork import NetworkObj
from pointunet import PointUNetObj
import random

parser = argparse.ArgumentParser()
parser.add_argument('--is_full', '-if', type=bool, default=False)
parser.add_argument('--pca_size', type=int, default=42)
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--save_dir', type=str, default='train_result', help="Folder to save pt file")
parser.add_argument('--model', '-m', type=int, default=1)

parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', '-e', type=int, default=50)

parser.add_argument('--device', '-d', type=str, default='cpu')
args = parser.parse_args()

now = datetime.now()
now_str = now.strftime("%d-%m-%Y-%H-%M-%S")
save_dir = os.path.join(args.save_dir, now_str)
os.makedirs(save_dir)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

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
train_dataset = DatasetObj(is_train=True, is_full=args.is_full, is_obj=True, device=device)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = DatasetObj(is_train=False, is_full=args.is_full, is_obj=True, device=device)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

logging.info("Train data: {}, Test data: {}".format(len(train_dataset), len(test_dataset)))
# logging.info("Train data: {}".format(len(train_dataset)))

### load model
network = None
if args.model == 1:
    network = NetworkObj()
elif args.model == 2:
    network = PointNet_Plus(args.ball_radius2)
elif args.model == 3:
    network = CascadedNetworkObj()
elif args.model == 4:
    network = PointUNetObj(args.ball_radius2)

network.to(device)
logging.info(network)

criterion = torch.nn.MSELoss(size_average=True).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas = (0.9, 0.999), eps=1e-05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
logging.info("================================================================================\n")

best_err = float("inf")
for epoch in range(args.epoch):
    ## training
    timer = time.time()
    train_mse = 0.0
    train_mse_wld = 0.0

    for i, data in enumerate(tqdm(train_dataloader, 0)):
        points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz = data

        ## compute output
        obb_len = torch.diff(bound_obb, dim=1)
        min_bound = bound_obb[:,:1,:]
        obb_center = min_bound + obb_len / 2

        estimation = None
        if isinstance(network, NetworkObj):
            estimation = network(points)
        elif isinstance(network, PointNet_Plus) or isinstance(network, PointUNetObj):
            inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
            estimation = network(inputs_level1, inputs_level1_center)
        elif isinstance(network, CascadedNetworkObj):
            estimation_stage_1, estimation_stage_2, estimation = network(points, train_dataset.pca_mean, train_dataset.pca_coeff)

        loss = None
        if isinstance(network, CascadedNetworkObj):
            loss = 0.25 * criterion(estimation_stage_1, gt_pca) + 0.25 * criterion(estimation_stage_2, gt_pca) + 0.5 * criterion(estimation, gt_pca)
        else:
            # loss = (0.8 * criterion(estimation[:, :63], gt_xyz) + 0.2 * criterion(estimation[:, 63:], obj_xyz.reshape(-1, 24))) * 1000

            # loss = criterion(
            #     torch.bmm(estimation.reshape(-1, 29, 3) * obb_len, volume_rotate),
            #     torch.bmm(torch.cat((gt_xyz, obj_xyz.reshape(-1, 24)), dim=1).reshape(-1, 29, 3) * obb_len, volume_rotate)
            # )

            obb_center_repeat = obb_center.squeeze_(1).repeat((1, 29))

            loss = (0.8 * criterion(estimation, torch.cat((gt_xyz, obj_xyz.reshape(-1, 24)), dim=1)) + \
                    0.2 * criterion(estimation, obb_center_repeat))

            # loss = 0.8 * criterion(
            #     torch.bmm(estimation[:, :63].reshape(-1, 21, 3) * obb_len, volume_rotate),
            #     torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len, volume_rotate)
            # ) + 0.2 * criterion(
            #     torch.bmm(estimation[:, 63:].reshape(-1, 8, 3) * obb_len, volume_rotate),
            #     torch.bmm(obj_xyz.reshape(-1, 8, 3) * obb_len, volume_rotate)
            # )
        
        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## update error
        train_mse = train_mse + loss.item()

        es_xyz_wld = torch.bmm(estimation.data[:, :63].reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
        gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
        train_mse_wld = train_mse_wld + criterion(es_xyz_wld, gt_xyz_wld)
    
    scheduler.step()
    
    logging.info("Time training: {} s".format(time.time() - timer))
    train_mse = train_mse / len(train_dataset)
    logging.info("MSE 1 sample: {} mm".format(train_mse))
    train_mse_wld = train_mse_wld / len(train_dataset)
    logging.info("Train error 1 sample in world space: {} mm".format(train_mse_wld))
    # logging.info("Epoch: {}, train error: {} mm".format(epoch, train_mse_wld))

    if best_err > train_mse:
        best_err = train_mse
        logging.info("Save best with error: {} mm".format(best_err))
        torch.save(network.state_dict(), os.path.join(save_dir, "network_best.pth".format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_best.pth".format(epoch)))

    # ## testing
    # timer = time.time()
    # test_mse = 0.0
    # test_mse_wld = 0.0

    # for i, data in enumerate(tqdm(test_dataloader, 0)):
    #     points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz = data

    #     obb_len = torch.diff(bound_obb, dim=1)
    #     min_bound = bound_obb[:,:1,:]

    #     ## compute output
    #     if isinstance(network, NetworkObj):
    #         estimation = network(points)
    #     elif isinstance(network, PointNet_Plus) or isinstance(network, PointUNetObj):
    #         inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
    #         estimation = network(inputs_level1, inputs_level1_center)
    #     elif isinstance(network, CascadedNetworkObj):
    #         estimation_stage_1, estimation_stage_2, estimation = network(points, train_dataset.pca_mean, train_dataset.pca_coeff)

    #     loss = None
    #     if isinstance(network, CascadedNetworkObj):
    #         loss = 0.25 * criterion(estimation_stage_1, gt_pca) + 0.25 * criterion(estimation_stage_2, gt_pca) + 0.5 * criterion(estimation, gt_pca)
    #     else:
    #         # loss = criterion(estimation, torch.cat((gt_xyz, obj_xyz.reshape(B, 24)), dim=1)) * 1000
    #         # loss = (0.8 * criterion(estimation[:, :63], gt_xyz) + 0.2 * criterion(estimation[:, 63:], obj_xyz.reshape(-1, 24))) * 1000

    #         # loss = criterion(
    #         #     torch.bmm(estimation.reshape(-1, 29, 3) * obb_len, volume_rotate),
    #         #     torch.bmm(torch.cat((gt_xyz, obj_xyz.reshape(-1, 24)), dim=1).reshape(-1, 29, 3) * obb_len, volume_rotate)
    #         # )

    #         loss = criterion(estimation, torch.cat((gt_xyz, obj_xyz.reshape(-1, 24)), dim=1)) * 1000

    #         # loss = 0.8 * criterion(
    #         #     torch.bmm(estimation[:, :63].reshape(-1, 21, 3) * obb_len, volume_rotate),
    #         #     torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len, volume_rotate)
    #         # ) + 0.2 * criterion(
    #         #     torch.bmm(estimation[:, 63:].reshape(-1, 8, 3) * obb_len, volume_rotate),
    #         #     torch.bmm(obj_xyz.reshape(-1, 8, 3) * obb_len, volume_rotate)
    #         # )

    #     ## update error
    #     test_mse = test_mse + loss.item()
    #     # pca_mean = train_dataset.pca_mean.expand(estimation.data.size(0), 63)
    #     # out_xyz = torch.addmm(pca_mean, estimation.data, train_dataset.pca_coeff)

    #     # obb_len = torch.diff(bound_obb, dim=1)
    #     # min_bound = bound_obb[:,:1,:]
    #     # out_xyz_wld = torch.bmm(estimation.data[:, :63].reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
    #     # gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
    #     # test_mse_wld = criterion(es_xyz_wld, gt_xyz_wld)
        
    # timer = (time.time() - timer) / len(test_dataset)
    # logging.info("Time test 1 sample: {} ms".format(timer * 1000))
    # test_mse = test_mse / len(test_dataset)
    # logging.info("Test MSE 1 sample: {} mm".format(test_mse))
    # # test_mse_wld = test_mse_wld / len(test_dataset)
    # # logging.info("Test error 1 sample in world space: {} mm".format(test_mse_wld))
    
    # logging.info("Epoch: {}, train error: {} mm, test error: {} mm".format(epoch, train_mse, test_mse))
    logging.info("================================================================================\n")