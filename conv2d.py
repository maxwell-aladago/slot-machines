from math import sqrt

import torch.nn.functional as F
from selection_engine import GreedySelection, ProbabilisticSelection
from torch import nn, empty, sum


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 k,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 greedy_selection=True
                 ):
        super(Conv2d, self).__init__()

        if isinstance(kernel_size, tuple):
            self.kH, self.kW = kernel_size
        elif isinstance(kernel_size, int):
            self.kH = self.kW = kernel_size
        else:
            print(type(kernel_size))
            raise ValueError("Kernel size must be an integer or a tuple of integers")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.k = k
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # weights are frozen throughout training
        self.weight = nn.Parameter(empty(out_channels, in_channels, self.kH, self.kW, k), requires_grad=False)
        self.score = nn.Parameter(empty(out_channels, in_channels, self.kH, self.kW, k))

        self.xavier_uniform_2d()

        if greedy_selection:
            self._selection_engine = GreedySelection.apply
        else:
            self._selection_engine = ProbabilisticSelection.apply

    def forward(self, x):
        selected_net = self._selection_engine(self.score)
        net_weight = sum(self.weight * selected_net, dim=-1)

        out = F.conv2d(input=x,
                       weight=net_weight,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       groups=self.groups
                       )
        return out

    def xavier_uniform_2d(self):
        fan_out = self.out_channels * self.kH * self.kW
        fan_in = self.in_channels * self.kH * self.kW
        std = sqrt(6.0 / (fan_in + fan_out))
        self.weight.data.uniform_(-std, std)
        self.score.data.uniform_(0, std)
