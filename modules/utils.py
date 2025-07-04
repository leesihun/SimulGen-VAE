import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch

def get_latest_file(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")

    latest_file = max(files, key=os.path.getmtime)
    return latest_file

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import time
class MyBaseDataset(Dataset):
    def __init__(self, x_data, load_all):
        print('Loading data...')
        if load_all:
            print('Loading all data...')
            self.x_data = torch.tensor(x_data)  # Store on CPU only
        else:
            self.x_data = x_data

    def __getitem__(self, index):
        output= self.x_data[index]
        return output

    def __len__(self):
        return len(self.x_data)

from skimage.util import random_noise
from torchvision.transforms import v2

class PINNDataset(Dataset):
    def __init__(self, x_data, y1_data, y2_data):
        self.x_data = torch.tensor(x_data)  # Store on CPU only
        self.y1_data = torch.tensor(y1_data)  # Store on CPU only
        self.y2_data = torch.tensor(y2_data)  # Store on CPU only

    def __getitem__(self, index):
        x = self.x_data[index]

        return x, self.y1_data[index], self.y2_data[index]
    
    def __len__(self):
        return self.x_data.shape[0]