from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMapEncoder(nn.Module):

    def __init__(
            self,
            map_channels: int,
            hidden_channels: Tuple[int, ...],
            output_size: int,
            masks: Tuple[int, ...],
            strides: Tuple[int, ...],
            patch_size: Tuple[int, ...],
            p_dropout: float = 0.5
    ):
        super().__init__()

        self.convolutions = nn.ModuleList()

        patch_size_x = patch_size[0] + patch_size[2]
        patch_size_y = patch_size[1] + patch_size[3]
        input_size = (map_channels, patch_size_x, patch_size_y)

        x_dummy = torch.ones(input_size).unsqueeze(0) \
            * torch.tensor(float('nan'))

        for i, hidden_size in enumerate(hidden_channels):
            self.convolutions.append(
                nn.Conv2d(
                    map_channels if i == 0 else hidden_channels[i-1],
                    hidden_channels[i],
                    masks[i],
                    strides[i]
                )
            )
            x_dummy = self.convolutions[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        for conv in self.convolutions:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.dropout(x)

        return x
