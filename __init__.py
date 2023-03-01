from .architecture import structured_embedding
import phenograph
import time
import torch
import random
import numpy as np
import pandas as pd
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys 
from .file_utils import *
from .df_utils import *
from .architecture import *
from .triplet_loss import *


#globals
sourceFile = ''
lambda_regul = 0
min_diff = 0.0
hard_loss = False
squared = False
triplet_margin = 0.2
device = torch.device('cpu')

def make_matrix_from_labels(labels):
    M = np.zeros((len(labels), len(labels)))
    for cluster in np.unique(labels):
        genes_in_cluster = np.where(labels == cluster)[0]
        for geneA in genes_in_cluster:
            for geneB in genes_in_cluster:
                M[geneA,geneB] = 1    
    return M

def train_model(model, optimizer, loader, label_x, label_y, epoch, lambda_super, train_name, train, semi, device):
    
    L_totals = []
    L_reconstruction_xs = []
    L_reconstruction_ys = []
    L_weights = []
    L_trip_batch_all_xs = []
    L_trip_batch_all_ys = []
    L_trip_batch_hard_xs = []
    L_trip_batch_hard_ys = []
    fraction_hard_xs = []
    fraction_hard_ys = []
    fraction_semi_xs = []
    fraction_semi_ys = []
    fraction_easy_xs =[]
    fraction_easy_ys =[]

    model.train()

    # loop over all batches
    for step, (batch_x_input, batch_y_input, batch_genes) in enumerate(loader):

        batch_label_x_input = label_x[batch_genes][:, batch_genes]
        batch_label_y_input = label_y[batch_genes][:, batch_genes]

        latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(batch_x_input, batch_y_input)     

        w_x = model.decoder_h_x.weight
        w_y = model.decoder_h_y.weight

        #calculate losses..

        #sparse penalty
        sparse_x = torch.sqrt(torch.sum(torch.sum(torch.square(w_x), axis=1)))
        sparse_y = torch.sqrt(torch.sum(torch.sum(torch.square(w_y), axis=1)))
        L_weight = sparse_x + sparse_y
        
        # triplet errors
        L_trip_batch_hard_x = batch_hard_triplet_loss(batch_label_x_input, latent, triplet_margin, semi, device, squared)
        L_trip_batch_hard_y = batch_hard_triplet_loss(batch_label_y_input, latent, triplet_margin, semi, device, squared)
        L_trip_batch_all_x, _ = batch_all_triplet_loss(batch_label_x_input, latent, triplet_margin, device, squared)
        L_trip_batch_all_y, _ = batch_all_triplet_loss(batch_label_y_input, latent, triplet_margin, device, squared)


        fraction_easy_x, fraction_semi_x, fraction_hard_x = fraction_triplets(batch_label_x_input, latent, triplet_margin, device)
        fraction_easy_y, fraction_semi_y, fraction_hard_y = fraction_triplets(batch_label_y_input, latent, triplet_margin, device)

        #reconstruction error
        L_reconstruction_x = torch.mean(torch.norm(reconstruct_x - batch_x_input))
        L_reconstruction_y = torch.mean(torch.norm(reconstruct_y - batch_y_input))
        
        L_weight = torch.mean(torch.square(latent))
        L_total = lambda_super*(L_trip_batch_all_x + L_trip_batch_all_y) +  lambda_regul*L_weight + L_reconstruction_x + L_reconstruction_y
        
        if hard_loss:
            L_total = lambda_super*(L_trip_batch_hard_x + L_trip_batch_hard_y) +  lambda_regul*L_weight + L_reconstruction_x + L_reconstruction_y

        if train == True:
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()


        L_totals.append(L_total.detach().cpu().numpy())
        L_reconstruction_xs.append(L_reconstruction_x.detach().cpu().numpy())
        L_reconstruction_ys.append(L_reconstruction_y.detach().cpu().numpy())
        L_weights.append(L_weight.detach().cpu().numpy())
        L_trip_batch_hard_xs.append(L_trip_batch_hard_x.detach().cpu().numpy())
        L_trip_batch_hard_ys.append(L_trip_batch_hard_y.detach().cpu().numpy())
        L_trip_batch_all_xs.append(L_trip_batch_all_x.detach().cpu().numpy())
        L_trip_batch_all_ys.append(L_trip_batch_all_y.detach().cpu().numpy())
        fraction_hard_xs.append(fraction_hard_x.detach().cpu().numpy())
        fraction_hard_ys.append(fraction_hard_y.detach().cpu().numpy())
        fraction_semi_xs.append(fraction_semi_x.detach().cpu().numpy())
        fraction_semi_ys.append(fraction_semi_y.detach().cpu().numpy())
        fraction_easy_xs.append(fraction_easy_x.detach().cpu().numpy())
        fraction_easy_ys.append(fraction_easy_y.detach().cpu().numpy())
        
    print( train_name+"_epoch:%d\ttotal_loss:%03.5f\treconstruction_loss_x:%03.5f\treconstruction_loss_y:%03.5f\tsparse_penalty:%03.5f\tx_triplet_loss_batch_hard:%03.5f\ty_triplet_loss_batch_hard:%03.5f\tx_triplet_loss_batch_all:%03.5f\ty_triplet_loss_batch_all:%03.5f\tx_fraction_hard:%03.5f\ty_fraction_hard:%03.5f\tx_fraction_semi:%03.5f\ty_fraction_semi:%03.5f\tx_fraction_easy:%03.5f\ty_fraction_easy:%03.5f"
        % (epoch, np.mean(L_totals), np.mean(L_reconstruction_xs), np.mean(L_reconstruction_ys), np.mean(L_weights), np.mean(L_trip_batch_hard_xs), np.mean(L_trip_batch_hard_ys), np.mean(L_trip_batch_all_xs), np.mean(L_trip_batch_all_ys), np.mean(fraction_hard_xs), np.mean(fraction_hard_ys),  np.mean(fraction_semi_xs), np.mean(fraction_semi_ys), np.mean(fraction_easy_xs), np.mean(fraction_easy_ys)), file = sourceFile)    
    

    
    
def muse_fit_predict(resultsdir, index, data_x,
                     data_y,
                     label_x = [],
                     label_y = [],
                     batch_size=64,
                     latent_dim=100,
                     n_epochs=500,
                     lambda_regul=5,
                     lambda_super=5, min_diff=0.2, hard_loss=False, squared=False, batch_norm=False, l2_norm = False, save_update_epochs=False, semi=False, k=30, dropout=0.25):
    """
        MUSE model fitting and predicting:
          This function is used to train the MUSE model on multi-modality data

        Parameters:
          resultsdir:   directory to save files
          index:        index names to use when saving files
          data_x:       input for transcript modality; matrix of  n * p, where n = number of cells, p = number of genes.
          data_y:       input for morphological modality; matrix of n * q, where n = number of cells, q is the feature dimension.
          label_x:      initial reference cluster label for transcriptional modality.
          label_y:      inital reference cluster label for morphological modality.
          latent_dim:   feature dimension of joint latent representation.
          n_epochs:     maximal epoch used in training.
          lambda_regul: weight for regularization term in the loss function.
          lambda_super: weight for supervised learning loss in the loss function.
          margin:       margin to use for triplet loss

        Output:
          latent:       joint latent representation learned by MUSE.
          reconstruct_x:reconstructed feature matrix corresponding to input data_x.
          reconstruct_y:reconstructed feature matrix corresponding to input data_y.
          latent_x:     modality-specific latent representation corresponding to data_x.
          latent_y:     modality-specific latent representation corresponding to data_y.

        Feng Bao @ Altschuler & Wu Lab @ UCSF 2022.
        Software provided as is under MIT License.
    """
    """ initial parameter setting """
    
    
    # parameter setting for neural network
    n_hidden = 128  # number of hidden node in neural network
    learn_rate = 1e-4  # learning rate in the optimization
    batch_size = 64  # number of cells in the training batch
    n_epochs_init = 200
    cluster_update_epoch = 50
    sourceFile = open('{}.txt'.format(resultsdir), 'w')
    
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name()
        
    # set globals  (same across all training)
    globals()['sourceFile'] = sourceFile
    globals()['lambda_regul'] = lambda_regul
    globals()['min_diff'] = min_diff
    globals()['hard_loss'] = hard_loss
    globals()['squared'] = squared
    globals()['device'] = device

    # read data-specific parameters from inputs
    feature_dim_x = data_x.shape[1]
    feature_dim_y = data_y.shape[1]
    n_sample = data_x.shape[0]
          
    # transform inputs to tensor
    transform=ToTensor()
    data_x = transform(data_x).to(device)
    data_y = transform(data_y).to(device)
    
    #make labels 
    create_label_x = False
    if len(label_x) == 0 :
        label_x, _, _ = phenograph.cluster(data_x.detach().cpu().numpy(), k=k)
        label_x = transform(make_matrix_from_labels(label_x)).to(device)
        create_label_x = True
    else:
        if (len(label_x.shape) == 1) or (label_x.shape[1] == 1) :
            label_x = transform(make_matrix_from_labels(label_x)).to(device)
        else:
            label_x = transform(label_x).to(device)
            
    create_label_y = False
    if len(label_y) == 0 :
        label_y, _, _ = phenograph.cluster(data_y.detach().cpu().numpy(), k=k)
        label_y = transform(make_matrix_from_labels(label_y)).to(device)
        create_label_y = True
    else:
        if (len(label_y.shape) == 1) or (label_y.shape[1] == 1) :
            label_y = transform(make_matrix_from_labels(label_y)).to(device)
        else:
            label_y = transform(label_y).to(device)
            
    # create model, optimizer, trainloader 
        
    model = structured_embedding(feature_dim_x, feature_dim_y, latent_dim, n_hidden, dropout, batch_norm, l2_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_loader = DataLoader(Protein_Dataset(data_x, data_y), batch_size=batch_size, shuffle=True)

     #INIT WITH JUST RECONSTRUCTION
    for epoch in range(n_epochs_init):
        model.train()
        train_model(model, optimizer, train_loader, label_x, label_y, epoch, 0, 'init_recon', True, semi, device)
        
  #  INIT WITH TRIPLET LOSS AND RECONSTRUCTION, ORIGINAL LABELS
    for epoch in range(n_epochs_init):
        model.train()
        train_model(model, optimizer, train_loader, label_x, label_y, epoch, lambda_super, 'init_both', True, semi, device)

    latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y) 
    
    update_label_x = label_x
    update_label_y = label_y
    if create_label_x:
        update_label_x, _, _ = phenograph.cluster(latent_x.detach().cpu().numpy(), k=k)
        update_label_x = transform(make_matrix_from_labels(update_label_x)).to(device)
    if create_label_y:
        update_label_y, _, _ = phenograph.cluster(latent_y.detach().cpu().numpy(), k=k)
        update_label_y = transform(make_matrix_from_labels(update_label_y)).to(device)
    
    # TRAIN WITH LABELS
    for epoch in range(n_epochs):
        model.train()
        train_model(model, optimizer, train_loader, update_label_x, update_label_y, epoch, lambda_super, 'train', True, semi, device)
        
        if epoch%cluster_update_epoch == 0:
            model.eval()
            with torch.no_grad():
                latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y)   
            

            if save_update_epochs:
                torch.save(model.state_dict(), '{}_{}.pth'.format(resultsdir, epoch))
                pd.DataFrame(latent.detach().cpu().numpy(), index = index).to_csv('{}_latent_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(reconstruct_x.detach().cpu().numpy(), index = index).to_csv('{}_reconstruct_x_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(reconstruct_y.detach().cpu().numpy(), index = index).to_csv('{}_reconstruct_y_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(latent_x.detach().cpu().numpy(), index = index).to_csv('{}_latent_x_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(latent_y.detach().cpu().numpy(), index = index).to_csv('{}_latent_y_{}.txt'.format(resultsdir, epoch))
            
            if create_label_x:
                update_label_x, _, _ = phenograph.cluster(latent_x.detach().cpu().numpy(), k=k)
                update_label_x = transform(make_matrix_from_labels(update_label_x)).to(device)
            if create_label_y:
                update_label_y, _, _ = phenograph.cluster(latent_y.detach().cpu().numpy(), k=k)
                update_label_y = transform(make_matrix_from_labels(update_label_y)).to(device)
                           
    #SAVE FINAL RESULTS
    model.eval()
    with torch.no_grad():
        latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y)
    
    torch.save(model.state_dict(), '{}.pth'.format(resultsdir))
    pd.DataFrame(latent.detach().cpu().numpy(), index = index).to_csv('{}_latent.txt'.format(resultsdir))
    pd.DataFrame(reconstruct_x.detach().cpu().numpy(), index = index).to_csv('{}_reconstruct_x.txt'.format(resultsdir))
    pd.DataFrame(reconstruct_y.detach().cpu().numpy(), index = index).to_csv('{}_reconstruct_y.txt'.format(resultsdir))
    pd.DataFrame(latent_x.detach().cpu().numpy(), index = index).to_csv('{}_latent_x.txt'.format(resultsdir))
    pd.DataFrame(latent_y.detach().cpu().numpy(), index = index).to_csv('{}_latent_y.txt'.format(resultsdir))
    
    return model