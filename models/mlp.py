import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import math
from utils import *


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    
    
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta 


class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


class MlpBlock(nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, mlp_output_dim,  activation='gelu', regularization=0):
        super().__init__()
        self.mlp_input_dim = mlp_input_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_output_dim = mlp_output_dim
        self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_output_dim)
        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == 'gelu':
            self.act1 = nn.GELU()
        else:
            raise ValueError('Unknown activation function type: %s'%activation)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.fc2(x)
        if self.reg2 is not None:
            x = self.reg2(x)
            
        return x


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim=26, channels_mlp_dim=512, regularization=0):
    #def __init__(self, tokens_mlp_dim=26, channels_mlp_dim=512, normal_dim=512, regularization=0):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.tokens_mlp_dim, self.tokens_mlp_dim, regularization=regularization)
        self.mlp_block_channel_mixing = MlpBlock(self.channels_mlp_dim, self.channels_mlp_dim, self.channels_mlp_dim, regularization=regularization)
        #self.mlp_block_channel_mixing = MlpBlock(normal_dim, normal_dim, self.channels_mlp_dim, regularization=regularization)

        self.se = SELayer(self.tokens_mlp_dim)

        self.LN1 = nn.LayerNorm(self.channels_mlp_dim)
        self.LN2 = nn.LayerNorm(self.channels_mlp_dim)
        #self.LN1 = nn.LayerNorm(normal_dim)
        #self.LN2 = nn.LayerNorm(normal_dim)
        #self.LN1 = Affine(self.channels_mlp_dim)
        #self.LN2 = Affine(self.channels_mlp_dim)
        
        
    def forward(self, x):
        y = self.LN1(x)
        
        y = y.transpose(1, 2)
        y = self.mlp_block_token_mixing(y)
        y = y.transpose(1, 2)
        
        y = self.se(y)
        x = x + y
                
        y = self.LN2(x)
        y = self.mlp_block_channel_mixing(y)  
        y = self.se(y)
            
        return x + y
        #return y


class MotionMLP(nn.Module):
    def __init__(self,
                 input_feats=16,
                 num_frames=25,
                 latent_dim=512,
                 num_layers=9,
                 dropout=0.2,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_feats = input_feats
        
        #dim = [512, 256, 128, 64, 32, 64, 128, 256, 512]
        #normal_dim = [512, 512, 256, 128, 64, 32, 64, 128, 256]
        
        #dim = [1024, 512, 256, 128, 64, 128, 256, 512, 1024]
        #normal_dim = [512, 1024, 512, 256, 128, 64, 128, 256, 512]

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.latent_dim)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim=self.num_frames+1,
                                                channels_mlp_dim=self.latent_dim,
                                                regularization=self.dropout)
                                     for i in range(self.num_layers)])
        #self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim=self.num_frames+1,
        #                                        channels_mlp_dim=dim[i],
        #                                        normal_dim=normal_dim[i],
        #                                        regularization=self.dropout)
        #                             for i in range(self.num_layers)])
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.cond_embed(mod.reshape(B, -1)).unsqueeze(1)
        h = self.joint_embed(x)
        h = torch.cat([emb, h], dim=1)
        
        for blk in self.blocks:
            h = blk(h)

        output = self.out(h[:, 1:, :]).view(B, T, -1).contiguous()
        return output