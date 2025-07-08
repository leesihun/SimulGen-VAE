from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def kl(mu, log_var):
    # Clamp log_var to prevent numerical instability
    log_var = torch.clamp(log_var, min=-10, max=10)
    var = torch.exp(log_var)
    loss = 0.5*torch.sum(mu**2+var-log_var-1, dim=[1])
    return torch.mean(loss, dim=0)

def kl_2(delta_mu, delta_log_var, mu, log_var):
    # Clamp log_var values to prevent numerical instability
    delta_log_var = torch.clamp(delta_log_var, min=-10, max=10)
    log_var = torch.clamp(log_var, min=-10, max=10)
    
    delta_var = torch.exp(delta_log_var)
    var = torch.exp(log_var)

    loss = -0.5*torch.sum(1+delta_log_var-delta_mu**2/var-delta_var, dim=[1,2])
    return torch.mean(loss, dim=0)

def log_sum_exp(x):
    m2 = torch.max(x, dim=1, keepdim=True)[0]
    m=m2.unsqueeze(1)
    return m+torch.log(torch.sum(torch.exp(x-m2), dim=1))
    

