import copy

import torch
from torch import nn


class SnakeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # convolutional layers for processing gaming board
        self.online_conv = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 2), stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 2 linear layers for processing the output of convolutional layers and 4 states for direction of snake
        self.online_linear = nn.Sequential(nn.Linear(117, 32), nn.ReLU(), nn.Linear(32, 4))

        self.target_conv = copy.deepcopy(self.online_conv)
        self.target_linear = copy.deepcopy(self.online_linear)

        for p in self.target_conv.parameters():
            p.requires_grad = False
        for p in self.target_linear.parameters():
            p.requires_grad = False

    def forward(self, state, model):
        board, direction = state
        if model == 'online':
            conv_out = self.online_conv(board)
            X = torch.concatenate((direction, conv_out), dim=1)
            return self.online_linear(X)
        elif model == 'target':
            conv_out = self.target_conv(board)
            X = torch.concatenate((direction, conv_out), dim=1)
            return self.target_linear(X)
