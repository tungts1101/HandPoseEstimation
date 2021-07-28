'''
author: Tran Son Tung
'''

from fpha_dataset import FPHADataset
from torch.utils.data import DataLoader

# ===== Load data =====
train_data = FPHADataset(root_path="point_cloud_dataset", train=True)