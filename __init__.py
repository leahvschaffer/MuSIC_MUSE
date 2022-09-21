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
from .triplet_loss import batch_hard_triplet_loss, batch_all_triplet_loss, fraction_triplets

""" Model fitting and feature prediction of MUSE """

def make_matrix_from_labels(labels):
    M = np.zeros((len(labels), len(labels)))
    for cluster in np.unique(labels):
        genes_in_cluster = np.where(labels == cluster)[0]
        for geneA in genes_in_cluster:
            for geneB in genes_in_cluster:
                M[geneA,geneB] = 1    
    return M

def muse_fit_predict(resultsdir, index, data_x,
                     data_y,
                     label_x,
                     label_y,
                     batch_size=64,
                     latent_dim=100,
                     n_epochs=500,
                     lambda_regul=5,
                     lambda_super=5, k=30, batch_all=False, save_update_epochs=False, triplet_margin=0.1):
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
          batch_all:    use batch all triplet loss 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.get_device_name()
    
    # parameter setting for neural network
    n_hidden = 128  # number of hidden node in neural network
    learn_rate = 1e-4  # learning rate in the optimization
    batch_size = 64  # number of cells in the training batch
    n_epochs_init = 200
    cluster_update_epoch = 20
    
    # read data-specific parameters from inputs
    feature_dim_x = data_x.shape[1]
    feature_dim_y = data_y.shape[1]
    n_sample = data_x.shape[0]

    #make matrix if labels are passed in as a 1D array
    if (len(label_x.shape) == 1) or (label_x.shape[1] == 1) :
        label_x = make_matrix_from_labels(label_x)
    if (len(label_y.shape) == 1) or (label_y.shape[1] == 1) :
        label_y = make_matrix_from_labels(label_y)
        
    model = structured_embedding(feature_dim_x, feature_dim_y, latent_dim, n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    transform=ToTensor()
    data_x = transform(data_x).to(device)
    data_y = transform(data_y).to(device)
    label_x = transform(label_x).to(device)
    label_y = transform(label_y).to(device)
    
    train_loader = DataLoader(Protein_Dataset(data_x, data_y, label_x, label_y), batch_size=batch_size, shuffle=True)
    sourceFile = open('{}.txt'.format(resultsdir), 'w')
    
    
    for epoch in range(n_epochs_init):
        
        model.train()

        # loop over all batches
        for step, (batch_x_input, batch_y_input, batch_genes) in enumerate(train_loader):
            
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

            #reconstruction error
            L_reconstruction_x = torch.mean(torch.norm(reconstruct_x - batch_x_input))
            L_reconstruction_y = torch.mean(torch.norm(reconstruct_y - batch_y_input))

            L_total = L_reconstruction_x + L_reconstruction_y + lambda_regul*L_weight

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            
    for epoch in range(n_epochs_init):
        
        model.train()

        # loop over all batches
        for step, (batch_x_input, batch_y_input, batch_genes) in enumerate(train_loader):
            
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
            L_trip_batch_hard_x = batch_hard_triplet_loss(batch_label_x_input, latent, triplet_margin)
            L_trip_batch_hard_y = batch_hard_triplet_loss(batch_label_y_input, latent, triplet_margin)

            #reconstruction error
            L_reconstruction_x = torch.mean(torch.norm(reconstruct_x - batch_x_input))
            L_reconstruction_y = torch.mean(torch.norm(reconstruct_y - batch_y_input))

            L_total = lambda_super*(L_trip_batch_hard_x + L_trip_batch_hard_y) + L_reconstruction_x + L_reconstruction_y + lambda_regul*L_weight

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            
    model.eval()
    with torch.no_grad():
        latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y)   

    update_label_x, _, _ = phenograph.cluster(latent_x.detach().cpu().numpy(), k=k)
    update_label_x = make_matrix_from_labels(update_label_x)
    
    update_label_y, _, _ = phenograph.cluster(latent_y.detach().cpu().numpy(), k=k)
    update_label_y = make_matrix_from_labels(update_label_y)
    
    train_loader = DataLoader(Protein_Dataset(data_x, data_y, update_label_x, update_label_y), batch_size=batch_size, shuffle=True)

    # refine MUSE parameters with reference labels and triplet losses
    ''' Training of MUSE '''
    for epoch in range(n_epochs):
        
        model.train()

        L_totals = []
        L_reconstruction_xs = []
        L_reconstruction_ys = []
        L_weights = []
        L_trip_batch_hard_xs = []
        L_trip_batch_hard_ys = []
        L_trip_batch_all_xs = []
        L_trip_batch_all_ys = []
        fraction_hard_xs = []
        fraction_hard_ys = []
        fraction_semi_xs = []
        fraction_semi_ys = []
        fraction_easy_xs =[]
        fraction_easy_ys =[]

        # loop over all batches
        for step, (batch_x_input, batch_y_input, batch_genes) in enumerate(train_loader):
            
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

            #reconstruction error
            L_reconstruction_x = torch.mean(torch.norm(reconstruct_x - batch_x_input))
            L_reconstruction_y = torch.mean(torch.norm(reconstruct_y - batch_y_input))

            # triplet errors
            L_trip_batch_hard_x = batch_hard_triplet_loss(batch_label_x_input, latent, triplet_margin)
            L_trip_batch_hard_y = batch_hard_triplet_loss(batch_label_y_input, latent, triplet_margin)

            
            L_trip_batch_all_x, _  = batch_all_triplet_loss(batch_label_x_input, latent, triplet_margin)
            L_trip_batch_all_y, _  = batch_all_triplet_loss(batch_label_y_input, latent, triplet_margin)
                        
            L_total = lambda_super*(L_trip_batch_hard_x + L_trip_batch_hard_y) + L_reconstruction_x + L_reconstruction_y + lambda_regul*L_weight
            
            if batch_all:
                L_total = lambda_super*(L_trip_batch_all_x + L_trip_batch_all_y) + L_reconstruction_x + L_reconstruction_y + lambda_regul*L_weight

            fraction_easy_x, fraction_semi_x, fraction_hard_x = fraction_triplets(batch_label_x_input, latent, triplet_margin)
            fraction_easy_y, fraction_semi_y, fraction_hard_y = fraction_triplets(batch_label_y_input, latent, triplet_margin)

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
    
        #print at the end of every epoch. Average loss/fractions over all batches. 
        print(
            "train_epoch:%d\ttotal_loss:%03.5f\treconstruction_loss_x:%03.5f\treconstruction_loss_y:%03.5f\tsparse_penalty:%03.5f\tx_triplet_loss_batch_hard:%03.5f\ty_triplet_loss_batch_hard:%03.5f\tx_triplet_loss_batch_all:%03.5f\ty_triplet_loss_batch_all:%03.5f\tx_fraction_hard:%03.5f\ty_fraction_hard:%03.5f\tx_fraction_semi:%03.5f\ty_fraction_semi:%03.5f\tx_fraction_easy:%03.5f\ty_fraction_easy:%03.5f"
            % (epoch, np.mean(L_totals), np.mean(L_reconstruction_xs), np.mean(L_reconstruction_ys), np.mean(L_weights), np.mean(L_trip_batch_hard_xs), np.mean(L_trip_batch_hard_ys), np.mean(L_trip_batch_all_xs), np.mean(L_trip_batch_all_ys), np.mean(fraction_hard_xs), np.mean(fraction_hard_ys),  np.mean(fraction_semi_xs), np.mean(fraction_semi_ys), np.mean(fraction_easy_xs), np.mean(fraction_easy_ys)), file = sourceFile)

        if epoch%cluster_update_epoch == 0:
            model.eval()
            with torch.no_grad():
                latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y)   
    
            update_label_x, _, _ = phenograph.cluster(latent_x.detach().cpu().numpy(), k=k)

            update_label_y, _, _ = phenograph.cluster(latent_y.detach().cpu().numpy(), k=k) 

            if save_update_epochs:
                torch.save(model.state_dict(), '{}_{}.pth'.format(resultsdir, epoch))
                pd.DataFrame(latent, index = index).to_csv('{}_latent_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(reconstruct_x, index = index).to_csv('{}_reconstruct_x_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(reconstruct_y, index = index).to_csv('{}_reconstruct_y_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(latent_x, index = index).to_csv('{}_latent_x_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(latent_y, index = index).to_csv('{}_latent_y_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(update_label_x, index = index).to_csv('{}_label_x_{}.txt'.format(resultsdir, epoch))
                pd.DataFrame(update_label_y, index = index).to_csv('{}_label_y_{}.txt'.format(resultsdir, epoch))
                        
                    
            update_label_x = make_matrix_from_labels(update_label_x)
            update_label_y = make_matrix_from_labels(update_label_y)
            train_loader = DataLoader(Protein_Dataset(data_x, data_y, update_label_x, update_label_y), batch_size=batch_size, shuffle=True)
        
    model.eval()
    with torch.no_grad():
        latent, reconstruct_x, reconstruct_y, latent_x, latent_y = model(data_x, data_y)
    
    torch.save(model.state_dict(), '{}.pth'.format(resultsdir))
    pd.DataFrame(latent, index = index).to_csv('{}_latent.txt'.format(resultsdir))
    pd.DataFrame(reconstruct_x, index = index).to_csv('{}_reconstruct_x.txt'.format(resultsdir))
    pd.DataFrame(reconstruct_y, index = index).to_csv('{}_reconstruct_y.txt'.format(resultsdir))
    pd.DataFrame(latent_x, index = index).to_csv('{}_latent_x.txt'.format(resultsdir))
    pd.DataFrame(latent_y, index = index).to_csv('{}_latent_y.txt'.format(resultsdir))