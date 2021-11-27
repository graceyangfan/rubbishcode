import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CNN(nn.Module):
    """
    Simple MLP network for one dimensional time series 
    All conv will keep seq_len the same 
    """

    def __init__(self, in_size: int,seq_len: int,mid_size, n_actions: int):
        """
        Args:
            input_shape: observation shape of the environment = [channel_seq_len]
            n_actions: number of discrete actions available in the environment
        """
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_size, mid_size, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(mid_size, mid_size, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv1d(mid_size,mid_size, kernel_size=3, padding =1 ),
            nn.ReLU(),
        )
        conv_out_size = mid_size * seq_len
        self.head = nn.Linear(conv_out_size,n_actions)

    def forward(self, input_x) -> Tensor:
        """
        Forward pass through network
        Args:
            input_x: time series with [batch,seq_len,feature_dim]
            =>[batch,feature_dim,seq_len]
        Returns:
            output of network
        """
        input_x = input_x.float()
        input_x = input_x.permute(0,2,1) 
        conv_out = self.conv(input_x).view(input_x.size()[0], -1)
        return self.head(conv_out)

