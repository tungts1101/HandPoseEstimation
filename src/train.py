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

from mydataset import DatasetObj
from mynetwork import NetworkObj
import random

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True, help="Root folder of FPHA dataset")
parser.add_argument('--is_full', type=bool, default=False)
parser.add_argument('--pca_size', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--save_dir', type=str, default='exp', required=True, help="Folder to save pt file")

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=50)

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
train_dataset = DatasetObj(root_path=args.root_path, is_train=True, is_full=args.is_full, device=device)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = DatasetObj(root_path=args.root_path, is_train=False, is_full=args.is_full, device=device)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

logging.info("Train data: {}, Test data: {}".format(len(train_dataset), len(test_dataset)))

### load model
network = NetworkObj()
network.to(device)
logging.info(network)

criterion = torch.nn.MSELoss(size_average=True).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas = (0.5, 0.999), eps=1e-06)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

for epoch in range(args.epoch):
    ## training
    timer = time.time()
    train_mse = 0.0
    train_mse_wld = 0.0

    for i, data in enumerate(tqdm(train_dataloader, 0)):
        points, gt_pca, gt_xyz, volume_rotate, bound_obb = data

        ## compute output
        optimizer.zero_grad()
        estimation = network(points)

        loss = criterion(estimation, gt_pca) * 42
        
        ## compute gradient
        # loss.backward()
        # optimizer.step()

        ## update error
        train_mse = train_mse + loss.item()*len(points)
        pca_mean = train_dataset.pca_mean.expand(estimation.data.size(0), 63)
        out_xyz = torch.addmm(pca_mean, estimation.data, train_dataset.pca_coeff)

        obb_len = torch.diff(bound_obb, dim=1)
        min_bound = bound_obb[:,:1,:]
        out_xyz_wld = torch.bmm(out_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
        gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)

        diff = torch.pow(out_xyz - gt_xyz, 2).view(-1, 21, 3)
        diff_sum_sqrt = torch.sqrt(torch.sum(diff, 2))
        diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
        train_mse_wld = train_mse_wld + diff_mean.sum()
    
    scheduler.step()
    
    logging.info("Time training: {} s".format(time.time() - timer))
    train_mse = train_mse / len(train_dataset)
    logging.info("Train error 1 sample: {} mm".format(train_mse))
    train_mse_wld = train_mse_wld / len(train_dataset)
    logging.info("Train error 1 sample in world space: {} cm".format(train_mse_wld))

    torch.save(network.state_dict(), os.path.join(save_dir, "network_{}.pth".format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_{}.pth".format(epoch)))

    ## testing
    timer = time.time()
    test_mse = 0.0
    test_mse_wld = 0.0

    for i, data in enumerate(tqdm(test_dataloader, 0)):
        points, gt_pca, gt_xyz, volume_rotate, bound_obb = data

        ## compute output
        estimation = network(points)
        loss = criterion(estimation, gt_pca) * 42

        ## update error
        test_mse = test_mse + loss.item()*len(points)
        pca_mean = train_dataset.pca_mean.expand(estimation.data.size(0), 63)
        out_xyz = torch.addmm(pca_mean, estimation.data, train_dataset.pca_coeff)

        obb_len = torch.diff(bound_obb, dim=1)
        min_bound = bound_obb[:,:1,:]
        out_xyz_wld = torch.bmm(out_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)
        gt_xyz_wld = torch.bmm(gt_xyz.reshape(-1, 21, 3) * obb_len + min_bound, volume_rotate)

        diff = torch.pow(out_xyz - gt_xyz, 2).view(-1, 21, 3)
        diff_sum_sqrt = torch.sqrt(torch.sum(diff, 2))
        diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
        test_mse_wld = test_mse_wld + diff_mean.sum()
        
    timer = (time.time() - timer) / len(test_dataset)
    logging.info("Time test 1 sample: {} ms".format(timer * 1000))
    test_mse = test_mse / len(test_dataset)
    logging.info("Test error 1 sample: {} mm".format(test_mse))
    test_mse_wld = test_mse_wld / len(test_dataset)
    logging.info("Test error 1 sample in world space: {} cm".format(test_mse_wld))

    logging.info("Epoch: {}, train error: {} cm, test error: {} cm".format(epoch, train_mse_wld, test_mse_wld))
    logging.info("================================================================================\n")