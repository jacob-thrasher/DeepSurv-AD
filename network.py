import torch
from torch import nn


class DeepSurv(nn.Module):
    def __init__(self, n_input: int, n_hidden_layers: int, hidden_dim: int, activation_fn='relu', dropout=0.5, do_batchnorm=False):
        '''
        Args:
            n_input - Number of input nodes
            n_hidden_layers - Number of hidden layerss
            hidden_dim - Number of nodes per hidden layer
            activation_fn - Nonlinearity (default relu)
            dropout - probability of dropout (default 0.5)
            do_batchnorm - Include batch normalization layers (default False)
        '''

        assert activation_fn in ['relu', 'selu'], f'Parameter "activation_fn" must be in [relu, selu], found {activation_fn}'

        self.do_batchnorm = do_batchnorm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_features=hidden_dim)

        if activation_fn == 'relu': self.activation_fn = nn.ReLU()
        else: activation_fn = nn.SELU()
        
        layers = []
        layers.append(nn.Linear(n_input, hidden_dim)) # Input layer
        for n in n_hidden_layers:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
    
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.do_batchnorm: x = self.norm(x)
            x = self.activation_fn(x)
            x = self.dropout(x)

        x = self.out(x)
        return x