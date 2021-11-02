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
from pointunet import PointUNetObj
import random

import json
import copy

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
parser.add_argument('--weight', '-w', type=str, help="Weight folder")
parser.add_argument('--fold', '-f', type=int, default=5, help="Number of folds")

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
        -1: {
            "global_best_err": 1e9
        },
        0: {
            "epoch": 1,
            "best_err": 1e9
        },
        1: {
            "epoch": 1,
            "best_err": 1e9
        },
        2: {
            "epoch": 1,
            "best_err": 1e9
        },
        3: {
            "epoch": 1,
            "best_err": 1e9
        },
        4: {
            "epoch": 1,
            "best_err": 1e9
        },
    }
    with open(os.path.join(save_dir, 'resume.json'), "w") as _to:
        json.dump(state, _to, indent=4)

with open(os.path.join(save_dir, 'resume.json'), "r") as _from:
    cur_state = json.load(_from)
    save_state = copy.deepcopy(cur_state)

device = torch.device('cuda:0') if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu'
logging.info("Device: {}".format(device))

### set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

### load data
train_dataset = DatasetObj(is_train=True, is_full=args.is_full, is_obj=True, device=device)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = DatasetObj(is_train=False, is_full=args.is_full, is_obj=True, device=device)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

k_folds = args.fold
dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

if not args.weight:
    # logging.info("Train data: {}, Test data: {}".format(len(train_dataset), len(test_dataset)))
    # logging.info("Train data: {}".format(len(train_dataset)))
    logging.info("Size data: {}".format(len(dataset)))

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

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
if not args.weight:
    logging.info(network)

criterion = torch.nn.MSELoss(size_average=True).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas = (0.9, 0.999), eps=1e-05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
logging.info("================================================================================\n")

results = {}
global_best_err = float(cur_state["-1"]['global_best_err'])

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    logging.info("Fold: {}".format(fold))
    logging.info("====================")

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_subsampler)

    network.apply(reset_weights)
    if args.weight != None:
        if os.path.exists(os.path.join(args.weight, "network_fold_{}_last.pth".format(fold))) and \
            os.path.exists(os.path.join(args.weight, "optimizer_fold_{}_last.pth".format(fold))):
            network.load_state_dict(torch.load(os.path.join(args.weight, "network_fold_{}_last.pth".format(fold))))
            optimizer.load_state_dict(torch.load(os.path.join(args.weight, "optimizer_fold_{}_last.pth".format(fold))))
        else:
            if os.path.exists(os.path.join(args.weight, "network_best.pth")) and \
                os.path.exists(os.path.join(args.weight, 'optimizer_best.pth')):
                network.load_state_dict(torch.load(os.path.join(args.weight, 'network_best.pth')))
                optimizer.load_state_dict(torch.load(os.path.join(args.weight, 'optimizer_best.pth')))

    best_err = float(cur_state[str(fold)]['best_err'])
    for epoch in range(int(cur_state[str(fold)]['epoch']), args.epoch + 1):
        logging.info("Epoch: {}".format(epoch))
        logging.info("====================")
        
        ## training
        timer = time.time()
        train_mse = 0.0
        train_mse_wld = 0.0

        for i, data in enumerate(tqdm(train_dataloader, 0)):
            points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz = data

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
                obb_len = torch.diff(bound_obb, dim=1)
                loss = criterion(estimation, torch.cat((gt_xyz, obj_xyz.reshape(-1, 24)), dim=1)) * 1000
                # loss = 0.9 * criterion(estimation[:, :63].reshape(-1, 21, 3) * obb_len, gt_xyz.reshape(-1, 21, 3) * obb_len) + \
                #     0.1 * criterion(estimation[:, 63:].reshape(-1, 8, 3) * obb_len, obj_xyz.reshape(-1, 8, 3) * obb_len)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## update error
            train_mse = train_mse + loss.item()

            # obb_len = torch.diff(bound_obb, dim=1)
            # min_bound = bound_obb[:,:1,:]
            # es_xyz_wld = torch.bmm(estimation.data[:, :63].reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
            # gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
            # train_mse_wld = train_mse_wld + criterion(es_xyz_wld, gt_xyz_wld)
        
        scheduler.step()
        
        logging.info("Time training 1 epoch: {} s".format(time.time() - timer))
        train_mse = train_mse / len(train_dataset)
        logging.info("Train error: {} mm".format(train_mse))
        logging.info("Epoch: {}, train error: {} mm".format(epoch, train_mse))
        # train_mse_wld = train_mse_wld / len(test_dataset)
        # logging.info("Train error in world space: {} mm".format(train_mse_wld))

        if best_err > train_mse:
            best_err = float(train_mse)
            logging.info("Save best error: {} mm".format(best_err))
            torch.save(network.state_dict(), os.path.join(save_dir, "network_best_fold_{}.pth".format(fold)))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_best_fold_{}.pth".format(fold)))

            save_state[str(fold)]["best_err"] = best_err
        
        if global_best_err > best_err:
            global_best_err = float(best_err)
            logging.info("Save global best error: {} mm".format(global_best_err))
            torch.save(network.state_dict(), os.path.join(save_dir, "network_best.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_best.pth"))

            save_state["-1"]["global_best_err"] = global_best_err
        
        torch.save(network.state_dict(), os.path.join(save_dir, "network_fold_{}_last.pth".format(fold)))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_fold_{}_last.pth".format(fold)))

        save_state[str(fold)]["epoch"] = epoch + 1 # for not training this epoch again

        with open(os.path.join(save_dir, 'resume.json'), "w") as _to:
            json.dump(save_state, _to, indent=4)

        if epoch % 10 == 0:
            ## testing
            timer = time.time()
            test_mse = 0.0
            test_mse_wld = 0.0

            with torch.no_grad():
                for i, data in enumerate(tqdm(test_dataloader, 0)):
                    points, gt_pca, gt_xyz, volume_rotate, bound_obb, obj_xyz = data

                    ## compute output
                    if isinstance(network, NetworkObj):
                        estimation = network(points)
                    elif isinstance(network, PointNet_Plus) or isinstance(network, PointUNetObj):
                        inputs_level1, inputs_level1_center = utils.group_points(points, args.ball_radius)
                        estimation = network(inputs_level1, inputs_level1_center)
                    elif isinstance(network, CascadedNetworkObj):
                        estimation_stage_1, estimation_stage_2, estimation = network(points, train_dataset.pca_mean, train_dataset.pca_coeff)

                    eval_loss = None
                    if isinstance(network, CascadedNetworkObj):
                        eval_loss = 0.25 * criterion(estimation_stage_1, gt_pca) + 0.25 * criterion(estimation_stage_2, gt_pca) + 0.5 * criterion(estimation, gt_pca)
                    else:
                        obb_len = torch.diff(bound_obb, dim=1)
                        eval_loss = criterion(estimation, torch.cat((gt_xyz, obj_xyz.reshape(-1, 24)), dim=1)) * 1000
                        # eval_loss = 0.9 * criterion(estimation[:, :63].reshape(-1, 21, 3) * obb_len, gt_xyz.reshape(-1, 21, 3) * obb_len) + \
                        #     0.1 * criterion(estimation[:, 63:].reshape(-1, 8, 3) * obb_len, obj_xyz.reshape(-1, 8, 3) * obb_len)

                    ## update error
                    test_mse = test_mse + eval_loss.item()

                    # obb_len = torch.diff(bound_obb, dim=1)
                    # min_bound = bound_obb[:,:1,:]
                    # out_xyz_wld = torch.bmm(estimation.data[:, :63].reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
                    # gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
                    # test_mse_wld = criterion(es_xyz_wld, gt_xyz_wld)
                
            timer = (time.time() - timer) / len(test_dataset)
            logging.info("Time test 1 sample: {} ms".format(timer * 1000))
            test_mse = test_mse / len(test_dataset)
            logging.info("Test error: {} mm".format(test_mse))
            # test_mse_wld = test_mse_wld / len(test_dataset)
            # logging.info("Test error in world space: {} mm".format(test_mse_wld))
    
    results[fold] = {
        "train_best_err": best_err,
        "test_err": test_mse
    }

    logging.info("================================================================================\n")

logging.info("K-Fold cross validation resuls for {} folds".format(k_folds))
sum_test_err = 0.0
for key, val in results.items():
    logging.info("Fold {}: {}".format(key, val))
    sum_test_err += val["test_err"]

logging.info("Average: {}".format(sum_test_err/k_folds))
logging.info("Global best error: {}".format(global_best_err))