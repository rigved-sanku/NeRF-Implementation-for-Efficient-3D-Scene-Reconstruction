import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os


class NeRFmodel(nn.Module): 
    
    def __init__(self, embedding_dim_pos = 10, embedding_dim_direction = 4, hidden_dim = 256, use_encoding = False):
        super(NeRFmodel, self).__init__()
        
        self.use_encoding = use_encoding
        
        if use_encoding:
            input_pos = (embedding_dim_pos * 6 + 3)
            input_dir = (embedding_dim_direction * 6 + 3)
        else:
            input_pos = 3
            input_dir = 3
        
        #print("input_pos: ", input_pos)  
        #print("input_dir: ", input_pos)  
        self.block1 = nn.Sequential(nn.Linear(input_pos, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        # Density estimation
        self.block2 = nn.Sequential(nn.Linear(input_pos + hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1))
        
        # color estimation
        self.block3 = nn.Sequential(nn.Linear(input_dir + hidden_dim, hidden_dim // 2), nn.ReLU())
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())
        
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()
                                    
    @staticmethod
    def position_encoding(x, num_encoding_functions, use_encoding = False):
        
        # Trivially, the input tensor is added to the posiiotntla encoding
        out = [x]
        
        # Now, encode the input using a set of high-frequency functions and append the
        # resulting values to the encoding.
        if use_encoding:
            for j in range(num_encoding_functions):
                out.append(torch.sin(2.0 ** j * x))
                out.append(torch.cos(2.0 ** j * x))
            out = torch.cat(out, dim = -1)
            return out
        
        else:
            return out[0]
    
    
    def forward(self, pos, direction):
        
        emb_x = self.position_encoding(pos, self.embedding_dim_pos, self.use_encoding) # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.position_encoding(direction, self.embedding_dim_direction, self.use_encoding) # emb_d: [batch_size, embedding_dim_direction * 6]
        
        #print("emb_x: ", emb_x.shape)
        
        out1 = self.block1(emb_x) # h: [batch_size, hidden_dim]
        out2 = self.block2(torch.cat([out1, emb_x], dim = -1)) # out2: [batch_size, hidden_dim + 1]
        
        out2, sigma = out2[:, :-1], self.relu(out2[:, -1]) # out2: [batch_size, hidden_dim], sigma: [batch_size]
        
        out3 = self.block3(torch.cat((out2, emb_d), dim=-1)) # out2: [batch_size, hidden_dim // 2]
        color = self.block4(out3) # color: [batch_size, 3]
        
        return color, sigma
        
        
    
