import torch
import torch.nn as nn
import numpy as np
from torchsummaryX import summary
import torch.nn.functional as F
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import natsort

def read_pinn_dataset_img(param_dir, param_data_type):
    cur_dir = os.getcwd()
    file_dir = cur_dir+param_dir

    im_size = 128

    if param_data_type == ".jpg" or param_data_type == ".png":
        print('Reading image dataset from '+file_dir+'\n')

        files = [f for f in os.listdir(file_dir) if f.endswith(param_data_type)]
        files = natsort.natsorted(files)

        pinn_data = np.zeros((len(files), im_size*im_size))
        i=0

        for file in files:
            print(file)
            file_path = file_dir+'/'+file
            im = cv2.imread(file_path, 0)

            resized_im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
            pinn_data[i, :] = resized_im.reshape(-1)[:]
            pinn_data_shape = resized_im.shape
            i=i+1
    else:
        raise NotImplementedError('Data type not supported')

    return pinn_data, pinn_data_shape

def read_pinn_dataset(param_dir, param_data_type): # For normal parametric approach: .csv
    pinn_data = pd.read_csv(param_dir, header=None)
    pinn_data = pinn_data.values

    return pinn_data


class PINN(nn.Module):
    def __init__(self, pinn_filter, latent_dim_end, input_shape, latent_dim, size2):
        super(PINN, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.pinn_filter = pinn_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_pinn_filter = len(self.pinn_filter)

        modules = []
        modules.append(nn.Linear(self.input_shape, self.pinn_filter[0]))
        for i in range(1, self.num_pinn_filter-1):
            modules.append(nn.Linear(self.pinn_filter[i-1], self.pinn_filter[i]))
            modules.append(nn.LeakyReLU(0.2))
            modules.append(nn.Dropout(0.2))
        self.pinn = nn.Sequential(*modules)

        self.latent_out = nn.Sequence(nn.Linear(self.pinn_filter[-2], self.pinn_filter[-1]),
                                      nn.LeakyReLU(0.2),
                                      nn.GroupNorm(1, self.pinn_filter[-1]),
                                      nn.Dropout(0.2),
                                      nn.Linear(self.pinn_filter[-1], self.latent_dim_end),
                                      nn.Tanh())

        self.xs_out = nn.Sequential(nn.Linear(self.pinn_filter[-2], self.pinn_filter[-1]),
                                    nn.LeakyReLU(0.2),
                                    nn.GroupNorm(1, self.pinn_filter[-1]),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.pinn_filter[-1], self.latent_dim*self.size2),
                                    nn.Unflatten(1, torch.size(self.size2, self.latent_dim)),
                                    nn.Tanh())

    def forward(self, x):
        x = self.pinn(x)
        latent_out = self.latent_out(x)
        xs_out = self.xs_out(x)

        return latent_out, xs_out



class ConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        multiple = 5

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel*multiple, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channel*multiple, out_channel, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        x = x+0.1*self.seq(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.seq(x)

class PINN_img(nn.Module):
    def __init__(self, pinn_filter, latent_dim_end, input_shape, latent_dim, size2, pinn_data_shape):
        super(PINN_img, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.pinn_filter = pinn_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_pinn_filter = len(self.pinn_filter)
        self.pinn_data_shape = pinn_data_shape

        modules = []
        modules.append(nn.Conv2d(1, self.pinn_filter[0], kernel_size=3, padding=1))
        modules.append(nn.GELU())
        for i in range(1, self.num_pinn_filter):
            modules.append(ConvResBlock(self.pinn_filter[i-1], self.pinn_filter[i-1]))
            modules.append(ConvBlock(self.pinn_filter[i-1], self.pinn_filter[i]))

        modules.append(nn.Flatten())
        modules.append(nn.LazyLinear(self.latent_dim_end*8))
        modules.append(nn.GELU())
        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(self.latent_dim_end*8, self.latent_dim_end))
        modules.append(nn.Tanh())

        self.latent_out = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Conv2d(1, self.pinn_filter[0], kernel_size=3, padding=1))
        for i in range(1, self.num_pinn_filter):
            modules.append(ConvResBlock(self.pinn_filter[i-1], self.pinn_filter[i-1]))
            modules.append(ConvBlock(self.pinn_filter[i-1], self.pinn_filter[i]))

        modules.append(nn.Flatten())
        modules.append(nn.LazyLinear(self.latent_dim_end*self.size2*8))
        modules.append(nn.GELU())
        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(self.latent_dim_end*self.size2*8, self.latent_dim_end*self.size2))
        modules.append(nn.Unflatten(1, torch.size(self.size2, self.latent_dim)))
        modules.append(nn.Tanh())

        self.xs_out = nn.Sequential(*modules)


    def forward(self, x):
        im_size = 128
        x = x.reshape(-1, 1, im_size, im_size)
        latent_out = self.latent_out(x)
        xs_out = self.xs_out(x)

        return latent_out, xs_out




import time
from modules.common import initialize_weights_He
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pytorch_warmup as warmup

def train_pinn(pinn_epoch, pinn_dataloader, pinn_validation_dataloader, pinn, pinn_lr):
    im_size = 128

    writer = SummaryWriter(log_dir = './PINNruns', comment = 'PINN')

    loss=0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pinn_optimized = torch.optim.AdamW(pinn.parameters(), lr=pinn_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pinn_optimized, T_max = pinn_epoch, eta_min = 0)

    from torchinfo import summary

    summary(pinn, (64,1,im_size,im_size))
    pinn = pinn.to(device)

    pinn.apply(initialize_weights_He)

    for epoch in range(pinn_epoch):
        start_time = time.time()
        pinn.train(True)
        for i, (x, y1, y2) in enumerate(pinn_dataloader):
            for param in pinn.parameters():
                param.grad = None

            y_pred1, y_pred2 = pinn(x)

            A = nn.MSELoss()(y_pred1, y1)
            B = nn.MSELoss()(y_pred2, y2)

            loss = A+B
            avgloss = loss

            if i==0:
                avg_loss_save = avgloss.detach().item()
            else:
                avg_loss_save = avg_loss_save+avgloss.detach().item()

            loss.backward()
            pinn_optimized.step()
        
        avg_loss_save = avg_loss_save/i

        for i, (x_val, y1_val, y2_val) in enumerate(pinn_validation_dataloader):
            pinn.eval()
            with torch.no_grad():
                y_pred1_val, y_pred2_val = pinn(x_val)

                A_val = nn.MSELoss()(y_pred1_val, y1_val)
                B_val = nn.MSELoss()(y_pred2_val, y2_val)

                val_loss = A_val+B_val

                if i==0:
                    val_loss_save = val_loss.detach().item()
                else:
                    val_loss_save = val_loss_save+val_loss.detach().item()

            val_loss_save = val_loss_save/i

            end_time = time.time()
            epoch_duration = end_time-start_time

            if epoch%10 == 0:
                writer.add_scalar('PINN Loss/train', avg_loss_save, epoch)
                writer.add_scalar('PINN Loss/val', val_loss_save, epoch)

            print('[%d/%d]\tPINN_Loss: %.4E, val_loss = %.4E, Main_Loss: %.4E, Hierarchical_loss: %.4E, ETA: %.4E h' % (epoch, pinn_epoch, avg_loss_save, val_loss_save, A, B, (pinn_epoch-epoch)*epoch_duration/3600))

            scheduler.step()

        torch.save(pinn.state_dict(), f'checpoints/pinn.pth')
        torch.save(pinn, 'model_save/PINN')

        return loss