# To run, type python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=1

"""
Source code for SimulGen-VAE
Developed by SiHun Lee, Ph. D., based on LSH-VAE
SimulGen-VAE is a pytorch implementation of LSH-VAE

Including image import version....

BOM
1. SimulGen-VAE input dataset: dataset#X.pickle: 3D array of [num_param, num_time, num_node]
2. PINN Input dataset: '.jpg, .png' files in file directory(for image, rest: .csv) -autodetect
3. preset.txt: preset file for SimulGen-VAE
4. input_data/condition.txt

To run, type the following command:
*** small
python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=1
for ETX Supercomputer, type the following command:
phd run -ng 1 -p shr_gpu -GR H100 -l %J.log python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=small --load_all=1

*** large
python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=large --load_all=1
for ETX Supercomputer, type the following command:
phd run -ng 1 -p shr_gpu -GR H100 -l %J.log python SimulGen-VAE.py --preset=1 --plot=2 --train_pinn_only=0 --size=large --load_all=1

*** Tensorboard
tensorboard --logdir=runs --port=6001
tensorboard --logdir=PINNruns --port=6002
"""

def main():
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np
    import argparse
    import matplotlib
    import matplotlib.pyplot as plt

    from modules.common import add_sn
    from modules.input_variables import input_user_variables, input_dataset
    from modules.data_preprocess import reduce_dataset, data_augmentation, data_scaler, pinn_scaler, pinn_scaler_input
    from modules.pinn import PINN_img, train_pinn, read_pinn_dataset_img, read_pinn_dataset, PINN
    from modules.plotter import temporal_plotter
    from modules.VAE_network import VAE
    from modules.train import train
    from modules.utils import MyBaseDataset, get_latest_file, PINNDataset

    from torchinfo import summary
    from torch.utils.data import DataLoader, Dataset, random_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", dest = "preset", action = "store")
    parser.add_argument("--plot", dest = "plot", action = "store")
    parser.add_argument("--train_pinn_only", dest = "train_pinn", action = "store")
    parser.add_argument("--size", dest = "size", action="store")
    parser.add_argument("--load_all", dest = "load_all", action = "store")
    args = parser.parse_args()

    def parse_condition_file(filepath):
        params = {}
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                # Remove comments and whitespace
                line = line.split('#')[0].strip()
                if not line or line.startswith('%') or line.startswith("'"):
                    continue  # skip empty lines and section markers
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    params[key] = value
        return params

    # Usage
    params = parse_condition_file('input_data/condition.txt')

    # Now you can access values by key, e.g.:
    num_param = int(params['Dim1'])
    num_time = int(params['Dim2'])
    num_time_to = int(params['Dim2_red'])
    num_node = int(params['Dim3'])
    num_node_to = int(params['Dim3_red'])
    num_var = int(params['num_var'])
    n_epochs = int(params['Training_epochs'])
    batch_size = int(params['Batch_size'])
    LR = float(params['LearningR'])
    latent_dim = int(params['Latent_dim'])
    latent_dim_end = int(params['Latent_dim_end'])
    loss_type = int(params['Loss_type'])
    stretch = int(params['Stretch'])
    alpha = int(params['alpha'])
    num_samples_f = int(params.get('num_aug_f', 0))
    num_samples_a = int(params.get('num_aug_a', 0))
    print_graph_recon = int(params.get('rec_graph', 0))
    recon_iter = int(params.get('Recon_iter', 1))
    num_physical_param = int(params['num_param'])
    n_sample = int(params['n_sample'])
    param_dir = params['param_dir']
    pinn_epoch = int(params['n_epoch'])
    pinn_lr = float(params['pinn_lr'])
    pinn_batch_size = int(params['pinn_batch'])
    pinn_data_type = params['input_type']
    param_data_type = params['param_data_type']

    input_shape = num_physical_param

    preset = args.preset
    size = args.size
    load_all_num = args.load_all
    if load_all_num=='1':
        load_all = True
    else:
        load_all = False

    print('load_all to GPU?', load_all)
    
    if size=='small':
        small=True
        print('Small network? :', small)
    elif size=='large':
        small=False
        print('Small network? :', small)
    else:
        NotImplementedError('Unrecoginized size arg')

    if preset =='1':
        with open('preset.txt') as f:
            lines = [line.rstrip('\n') for line in f]

        data_No = int(lines[1])
        init_beta_diviser = int(lines[2])
        num_filter_enc = list(map(int, lines[3].split()))
        pinn_filter = list(map(int, lines[4].split()))
    else:
        data_No, init_beta_divisor, num_filter_enc, pinn_filter = input_user_variables()

    if loss_type == 1:
        loss = 'MSE'
    if loss_type == 2:
        loss = 'MAE'
    if loss_type == 3:
        loss = 'smoothL1'
    if loss_type == 4:
        loss = 'Huber'

    init_beta = 10**(-1*init_beta_diviser)

    num_filter_dec = num_filter_enc[::-1]
    strides = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    num_layer_enc = len(num_filter_enc)
    num_layer_dec = len(num_filter_dec)

    num_node_red_start = 0
    num_node_red_end = num_node
    num_node_red = num_node_red_end - num_node_red_start
    initial_learning_rate = LR
    data_division = 1
    warm_up_rate = 1

    # Display lots of data

    print_graph = args.plot

    train_pinn_only = int(args.train_pinn)

    data_save = input_dataset(num_param, num_time, num_node, data_No)
    num_time, FOM_data, num_node = reduce_dataset(data_save, num_time_to, num_node_red, num_param, num_time, num_node_red_start, num_node_red_end)
    del data_save

    FOM_data_aug = data_augmentation(stretch, FOM_data, num_param, num_node)
    #temporal_plotter(FOM_data_aug, 0, 7,0,print_graph,7)
    new_x_train, _, _ = data_scaler(FOM_data_aug, FOM_data, num_time, num_node, data_No)

    del FOM_data, FOM_data_aug

    #pytorch: [batch_size, num_channels, seqe_length]
    new_x_train = new_x_train.transpose((0,2,1))
    new_x_train = np.float32(new_x_train)

    print('Dataset reange: ', np.min(new_x_train), np.max(new_x_train))
    
    dataset = MyBaseDataset(new_x_train, load_all)
    train_dataset, validation_dataset = random_split(dataset, [int(0.8*num_param), num_param - int(0.8*num_param)])

    if load_all:
        dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle =True, num_workers = 0, pin_memory = False, drop_last = True)
        val_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle =True, num_workers = 0, pin_memory = False, drop_last = True)
    else:
        dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle =True, num_workers = 0, pin_memory = True, drop_last = True)
        val_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle =True, num_workers = 0, pin_memory = True, drop_last = True)
    
    del train_dataset, validation_dataset

    print('Dataloader initiated...')

    # VAE training
    if train_pinn_only ==0:

        VAE_loss, reconstruction_error, KL_divergence, loss_val_print = train(n_epochs, batch_size, dataloader, val_dataloader, LR, num_filter_enc, num_filter_dec, num_node, latent_dim_end, latent_dim, num_time, alpha, loss, small, load_all)
        del dataloader, val_dataloader

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        VAE_trained = torch.load('./model_save/SimulGen-VAE', map_location= device, weights_only=False)
        VAE = VAE_trained.eval()

        from modules.decoder import reparameterize

        latent_vectors = np.zeros([num_param, latent_dim_end])
        gen_x_node = np.zeros([1, num_node, num_time])
        loss_total = 0
        loss_save = np.zeros([num_var])
        reconstruction_loss = np.zeros([num_param])
        hierarchical_latent_vectors = np.zeros([num_param, len(num_filter_enc)-1, latent_dim])
        reconstructed = np.empty([num_param, num_node, num_time])
        dataloader2 = DataLoader(dataset, batch_size = 1, shuffle =False, num_workers = 0, pin_memory = True, drop_last = False)

        for j, image in enumerate(dataloader2):
            loss_save[:]=100
            x = image.to(device)
            del image

            mu, log_var, xs = VAE.encoder(x)
            for i in range(recon_iter):
                std = torch.exp(0.5*log_var)
                latent_vector = reparameterize(mu, std)

                gen_x, _ = VAE.decoder(latent_vector, xs, mode='random')
                gen_x_np = gen_x.cpu().detach().numpy()

                loss = nn.MSELoss()(gen_x, x)

                if loss<loss_save[0]:
                    loss_save[0] = loss
                    latent_vector_save = latent_vector
                    latent_vectors[j,:] = latent_vector_save[0,:].cpu().detach().numpy()

                    for k in range(len(xs)):
                        hierarchical_latent_vectors[j,k,:] = xs[k].cpu().detach().numpy()[0]

                    reconstruction_loss[j] = loss

                    reconstructed[j,:,:] = gen_x_np[0,:,:]

                    del latent_vector, x, mu, log_var, xs, std, gen_x, gen_x_np, latent_vector_save

            print('parameter {} is finished''-''MSE: {:.4E}'.format(j+1, loss))
            print('')
            loss_total = loss_total+loss.cpu().detach().numpy()

            del loss

        print('')

        print('Total MSE loss: {:.3e}'.format(loss_total/num_param))

        np.save('model_save/latent_vectors', latent_vectors)
        np.save('model_save/xs', hierarchical_latent_vectors)

        temp5 = './SimulGen-VAE_L2_loss.txt'
        np.savetxt(temp5, reconstruction_loss, fmt = '%e')

        if print_graph_recon == 1:
            print('Printing graph...')
            plt.semilogy(VAE_loss, label = 'VAE')
            plt.semilogy(loss_val_print, label = 'Validation')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(reconstruction_loss, label = 'Reconstruction')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            plt.title('Nodal data')
            plt.plot(dataset[param_No,:,0].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,:,0], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 10
            plt.title('Nodal data')
            plt.plot(dataset[param_No,:,0].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,:,0], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            node_No = 900
            plt.title('Temporal data')
            plt.plot(dataset[param_No,node_No,:].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,node_No,:], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            node_No = 800
            plt.title('Temporal data')
            plt.plot(dataset[param_No,node_No,:].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,node_No,:], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            node_No = 700
            plt.title('Parametric data')
            plt.plot(dataset[param_No, node_No, :].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.plot(reconstructed[param_No+1, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.plot(reconstructed[param_No+2, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.plot(reconstructed[param_No+3, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

        elif train_pinn_only == 1:
            print('Training PINN only...')
            latent_vectors = np.load('model_save/latent_vectors.npy')
            hierarchical_latent_vectors = np.load('model_save/xs.npy')
            device = "cpu"
            VAE_trained = torch.load('model_save/SimulGen-VAE', map_location= device, weights_only=False)
            VAE = VAE_trained.eval()

        else:
            raise Exception('Unrecoginized train_pinn_only arg')

        out_latent_vectors = latent_vectors.reshape([num_param, latent_dim_end])
        xs_vectors = hierarchical_latent_vectors.reshape([num_param, -1])


        if pinn_data_type=='image':
            pinn_data, pinn_data_shape = read_pinn_dataset_img(param_dir, param_data_type)
        elif pinn_data_type=='csv':
            pinn_data = read_pinn_dataset(param_dir, param_data_type)
        else:
            NotImplementedError('Unrecoginized pinn_data_type arg')

        physical_param_input = pinn_data

        physical_param_input, param_input_scaler = pinn_scaler(physical_param_input, './model_save/pinn_input_scaler.pkl')
        out_latent_vectors, latent_vectors_scaler = pinn_scaler(out_latent_vectors, './model_save/latent_vectors_scaler.pkl')
        out_hierarchical_latent_vectors, xs_scaler = pinn_scaler(xs_vectors, './model_save/xs_scaler.pkl')

        print('PINN data loaded...')
        print('PINN scale: ')
        print(f'Input: {np.max(physical_param_input)} {np.min(physical_param_input)}')
        print(f'Main latent: {np.max(out_latent_vectors)} {np.min(out_latent_vectors)}')
        print(f'Hierarchical latent: {np.max(out_hierarchical_latent_vectors)} {np.min(out_hierarchical_latent_vectors)}')
        out_hierarchical_latent_vectors = out_hierarchical_latent_vectors.reshape([num_param, len(num_filter_enc)-1, latent_dim])

        pinn_dataset = PINNDataset(np.float32(physical_param_input), np.float32(out_latent_vectors), np.float32(out_hierarchical_latent_vectors))

        pinn_train_dataset, pinn_validation_dataset = random_split(pinn_dataset, [int(0.8*num_param), num_param - int(0.8*num_param)])
        pinn_dataloader = torch.utils.data.Dataloader(pinn_train_dataset, batch_size = pinn_batch_size, shuffle=True, num_workers = 0)
        pinn_validation_dataloader = torch.utils.data.Dataloader(pinn_validation_dataset, batch_size = pinn_batch_size, shuffle=False, num_workers = 0)

        size2 = len(num_filter_enc)-1


        if pinn_data_type=='image':
            pinn = PINN_img(pinn_filter, latent_dim_end, input_shape, latent_dim, size2, pinn_data_shape).to(device)
        elif pinn_data_type=='csv':
            pinn = PINN(pinn_filter, latent_dim_end, input_shape, latent_dim, size2).to(device)
        else:
            NotImplementedError('Unrecoginized pinn_data_type arg')

        print(pinn)

        PINN_loss = train_pinn(pinn_epoch, pinn_dataloader, pinn_validation_dataloader, pinn, pinn_lr)

        latent_x = np.linspace(0, latent_dim_end-1, latent_dim_end)
        latent_hierarchical_x = np.linspace(0, latent_dim-1, latent_dim)

        device = "cpu"
        pinn = pinn.to(device)

        if VAE in globals():
            del VAE

        VAE_trained = torch.load('model_save/SimulGen-VAE', map_location= device, weights_only=False)
        VAE = VAE_trained.eval()

        pinn_dataloader_eval = torch.utils.data.Dataloader(pinn_dataset, batch_size = 1, shuffle=False, num_workers = 0)

        for i, (x, y1, y2) in enumerate(pinn_dataloader_eval):
            x = x.to(device)

            y_pred1, y_pred2 = pinn(x)
            y_pred1 = y_pred1.cpu().detach().numpy()
            y1 = y1.cpu().detach().numpy()
            y_pred2 = y_pred2.cpu().detach().numpy()
            y2 = y2.cpu().detach().numpy()
            A = y2

            y2 = y2.reshape([1, -1])
            latent_predict = latent_vectors_scaler.inverse_transform(y1)
            xs_predict = xs_scaler.inverse_transform(y2)
            xs_predict = xs_predict.reshape([-1, 1, A.shape[-1]])
            latent_predict = torch.from_numpy(latent_predict)
            xs_predict = torch.from_numpy(xs_predict)
            xs_predict = xs_predict.to(device)
            xs_predict = list(xs_predict)
            latent_predict = latent_predict.to(device)
            target_output, _ = VAE.decoder(latent_predict, xs_predict, mode='fix')
            target_output_np = target_output.cpu().detach().numpy()
            target_output_np = target_output_np.swapaxes(1,2)
            target_output_np = target_output_np.reshape((-1, num_node))
            target_output_np = np.reshape(target_output_np, [num_time, num_node, 1])

            plt.figure()
            plt.title('Main latent')
            plt.plot(y1[0,:], '*', label = 'True')
            plt.plot(y_pred1[0,:], 'o', label = 'Predicted')
            plt.legend()

            plt.figure()
            plt.title('Hierarchical latent')
            plt.plot(y2[0,:], '*', label = 'True')
            plt.plot(y_pred2[0,0, :], 'o', label = 'Predicted')
            plt.legend()
            
            plt.figure()
            plt.title('Reconstruction')
            plt.plot(target_output_np[0,:,0], '.', label = 'Recon')
            plt.plot(new_x_train[i, :, int(num_time/2)], '.', label = 'True')
            plt.legend()

            plt.show()

if __name__ == '__main__':
    main()
