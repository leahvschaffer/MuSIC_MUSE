### Classes used for coembedding 
import torch
import random
import numpy as np
import pandas as pd
import math
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances, cosine_similarity
from .triplet_loss import batch_hard_triplet_loss

def cos_sim(A, B):
        cosine = np.dot(A,B)/(norm(A)*norm(B))
        return cosine

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        return torch.from_numpy(sample).float()
              
    
class Protein_Dataset(Dataset):
    def __init__(self, data_train_x, data_train_y):
        self.data_train_x = data_train_x
        self.data_train_y = data_train_y
                
    def __len__(self):
        return len(self.data_train_x)
    
    def __getitem__(self, item):
        return self.data_train_x[item], self.data_train_y[item], item

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
     #   nn.init.kaiming_normal(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)

def init_weights_d(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data)
            
class structured_embedding(nn.Module):
    def __init__(self, x_input_size, y_input_size, latent_dim, hidden_size, dropout, batch_norm, l2_norm):
        super().__init__()
                
        self.batch_norm = batch_norm 
        self.l2_norm = l2_norm
        
        self.encoder_x = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(x_input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh())
    
        self.encoder_y = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(y_input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh())
        
        self.encoder_z = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, latent_dim))
         #   nn.BatchNorm1d(latent_dim))
        
        self.decoder_h_x = nn.Linear(latent_dim, latent_dim, bias=False)
        self.decoder_h_y = nn.Linear(latent_dim, latent_dim, bias=False)
        
        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, x_input_size))
        
        self.decoder_y = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, y_input_size))
       
        # initialize weights
        self.encoder_x.apply(init_weights)
        self.encoder_y.apply(init_weights)
        self.encoder_z.apply(init_weights)
        self.decoder_h_x.apply(init_weights_d)
        self.decoder_h_y.apply(init_weights_d)
        self.decoder_x.apply(init_weights)
        self.decoder_y.apply(init_weights)
        
        self.batchnorm = nn.BatchNorm1d(latent_dim)
       
    
    def forward(self, x, y):
        
        h_x = self.encoder_x(x)
        h_y = self.encoder_y(y)
            
        h = torch.cat((h_x, h_y), 1)
        z = self.encoder_z(h)
        
        #unit sphere
        if self.batch_norm:
            z = self.batchnorm(z)
        if self.l2_norm:
            z = nn.functional.normalize(z, p=2, dim=1)

        z_x = self.decoder_h_x(z)
        z_y = self.decoder_h_y(z)
        
        x_hat = self.decoder_x(z_x)
        y_hat = self.decoder_y(z_y)


        return z, x_hat, y_hat, h_x, h_y
        
        
        


         