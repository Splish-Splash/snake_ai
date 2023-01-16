import torch
from torch import nn
from config import *
d = torch.ones((1, 3, NUMBER_OF_CELLS, NUMBER_OF_CELLS), dtype=torch.float32)

model = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 2), stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

linear = nn.Sequential(nn.Linear(72 + 4, 32), nn.ReLU(), nn.Linear(32, 4))
directions = torch.zeros((1, 4), dtype=torch.float32)
conv_out = model(d)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
X = torch.concatenate((directions, conv_out), dim=1)

print(linear(X))