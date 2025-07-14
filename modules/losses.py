from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def kl(mu, log_var):
    var = torch.exp(log_var)
    loss = 0.5*torch.sum(mu**2+var-log_var-1, dim=[1])
    return torch.mean(loss, dim=0)

def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)
    
    loss = 0.5*torch.sum(torch.div(delta_var, var)+torch.div((mu-delta_mu)**2, var)-delta_log_var+log_var-1, dim=[1,2])
    return torch.mean(loss, dim=0)

def log_sum_exp(x):
    m2 = torch.max(x, dim=1, keepdim=True)[0]
    m=m2.unsqueeze(1)
    return m+torch.log(torch.sum(torch.exp(x-m2), dim=1))
