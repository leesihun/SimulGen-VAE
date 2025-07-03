import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
from audiomentations import *

def reduce_dataset(data_save, num_time_to, num_node_red, num_param, num_time, num_node_red_start, num_node_red_end):
    import time

    start = time.time()

    num_node = data_save.shape[-1]
    if num_time_to == num_time and num_node_red == num_node:
        FOM_data = data_save
    else:
        FOM_data_temp = data_save
        num_time = num_time_to
        FOM_data = np.zeros((num_param, num_time, num_node_red))
        FOM_data[:, 0:num_time,:] = FOM_data_temp[:, 0:num_time, num_node_red_start:num_node_red_end]
        del FOM_data_temp

        num_node = num_node_red
        FOM_data_temp = np.zeros((num_param, num_time, num_node_red))
        FOM_data_temp[:,:,0:num_node] = FOM_data
        FOM_data_temp[:,:,num_node:-1] = 0
        del FOM_data

        FOM_data = np.zeros((num_param, num_time, num_node))
        FOM_data = FOM_data_temp

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print('FOM shape:   ', FOM_data.shape)

    return num_time, FOM_data, num_node

def data_augmentation(stretch, FOM_data, num_param, num_node):
    #### T.B.D. w. audiomentation, librosa
    # Not currently used at the moment

    if stretch  == 1:
        new_x_train = FOM_data

        augment = Compose([
            AddGaussianNoise(min_amplitude = 0.001, max_amplitude = 0.05, p=1),
            Resample(min_sample_rate = 1000, max_sample_rate = 15000, p=1),
            Shift(p=1),
        ])

        for i in range(num_param):
            X = FOM_data[i, :,:]
            X = augment(samples = X, sample_rate = 10000)
            new_x_train[i,:,0:num_node] = X[:,0:num_node]

        FOM_data_aug = np.append(FOM_data, new_x_train, axis=0)

    else:
        FOM_data_aug = FOM_data

    return FOM_data_aug

def data_scaler(FOM_data_aug, FOM_data, num_time, num_node, directory):
    import time

    start= time.time()

    scaler = MinMaxScaler(feature_range=(-0.7, 0.7))

    x_train_temp = FOM_data_aug.reshape([-1, num_node])
    FOM_temp = FOM_data.reshape([-1, num_node])

    scaler.fit(x_train_temp)
    x_train_temp = scaler.transform(x_train_temp)
    scaled_FOM = scaler.transform(FOM_temp)
    x_train_temp = x_train_temp.reshape([-1, num_time, num_node])

    x_train = x_train_temp
    x_train.shape
    new_x_train = x_train
    DATA_shape = new_x_train.shape[1:]

    dump(scaler, open('./model_save/scaler.pkl', 'wb'))
    end= time.time()
    print(f"Scaling time: {end - start} seconds")

    return new_x_train, DATA_shape, scaler

def pinn_scaler(data, name):
    scaler = MinMaxScaler(feature_range=(-0.7, 0.7))

    scaler.fit(data)
    scaled_data = scaler.transform(data)

    dump(scaler, open(name, 'wb'))

    return scaled_data, scaler

def pinn_scaler_input(data, name):

    scaler = StandardScaler()

    scaler.fit(data)
    scaled_data = scaler.transform(data)

    dump(scaler, open(name, 'wb'))

    return scaled_data, scaler