import numpy as np
import torch
from torch import nn


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(
            np.int32
        )
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=tuple(kernel_size), stride=tuple(stride_size))
        x = avg(x)
        return x
