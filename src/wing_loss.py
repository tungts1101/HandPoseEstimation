import torch
from torch import nn
import math

class WingLoss(nn.Module):
    def __init__(self, omega, epsilon):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * math.log((1+omega/epsilon))
    
    def forward(self, y_hat,y):
        diff = y_hat - y.view(y.shape[0],-1)
        diff_abs = diff.abs()
        loss = diff_abs.clone()
        
        idx_smaller = diff_abs < self.omega
        idx_bigger = diff_abs >= self.omega
        
        loss[idx_smaller] = self.omega * torch.log(1 + loss[idx_smaller].abs() / self.epsilon)
        loss[idx_bigger] = loss[idx_bigger].abs() - self.C
        loss = loss.mean()
        
        return loss
