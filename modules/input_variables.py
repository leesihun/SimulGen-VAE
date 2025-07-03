def input_user_variables():
    import pandas as pd
    import numpy as np
    
    print('Input dataset number of pickle file, dataset%d.pickle')
    datasetNo = int(input())

    print('Input initial beta power')
    init_beta_divisior = int(input())

    print('Input SimulGen-VAE filters')
    num_filter_enc = list(map(int, input().split()))

    print('Input PINN filters')
    pinn_filter = list(map(int, input().split()))

    return (datasetNo, init_beta_divisior, num_filter_enc, pinn_filter)

def show_graph():
    print('Show graph? 1 for yes, 2 for no')
    print_graph = int(input())

    return print_graph

def input_dataset(num_param, num_time, num_node, data_No):
    import numpy as np
    import pickle
    import time

    start= time.time()
    data_save = np.zeros((num_param, num_time, num_node))\

    print('Opening dataset file: dataset%d.pickle' % data_No)

    with open('dataset%d.pickle' % data_No, 'rb') as fr:
        data_save = pickle(load(fr))

    end = time.time()
    print('Time taken to load dataset: %f seconds' % (end-start))
    print('Dataset size: ', data_save.shape)

    return data_save