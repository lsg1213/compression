import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class VINR(nn.Module):
    '''
    Assumptions
        1. mono audio
    '''
    def __init__(self, in_dim, out_dim, n_hidden_layers=3,
                 hidden_dim=64, activation='gelu', n_bits=8):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers

        net = [QALinear(in_dim, hidden_dim, n_bits=n_bits)]
        # net = [nn.Linear(in_dim, hidden_dim, bias=bias)]

        for i in range(n_hidden_layers):
            net.append(get_activation_fn(activation))
            net.append(QALinear(hidden_dim, hidden_dim, n_bits=n_bits))
            # net.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))

        net.extend([get_activation_fn(activation),
                    QALinear(hidden_dim, out_dim, n_bits=n_bits)])
                    # nn.Linear(hidden_dim, out_dim, bias=bias))])

        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        return torch.tanh(self.net(inputs))

    def get_bit_size(self):
        return sum([0 if not isinstance(l, QALinear) else l.get_bit_size()
                    for l in self.net])


class QALinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_bits=8, quant_axis=(-2, -1)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_bits = n_bits
        self.quant_axis = quant_axis

        self.weight = nn.Parameter(
            2 * torch.rand(in_dim, out_dim) / np.sqrt(in_dim)
            - 1 / np.sqrt(in_dim), requires_grad=True)
        self.bias = nn.Parameter(
            2 * torch.rand(out_dim) / np.sqrt(in_dim)
            - 1 / np.sqrt(in_dim), requires_grad=True)

    def forward(self, inputs):
        # quantize
        r_weight = self.rounding(self.weight, self.quant_axis)
        weight = (r_weight - self.weight).detach() + self.weight

        bias = (self.rounding(self.bias, -1) - self.bias).detach() + self.bias

        return inputs @ weight + bias

    def rounding(self, inputs, axis):
        min_value = torch.amin(inputs, axis, keepdims=True)
        max_value = torch.amax(inputs, axis, keepdims=True)
        scale = (max_value - min_value) / (self.n_bits**2 - 1)

        return torch.round((inputs - min_value) / scale) * scale + min_value

    def get_bit_size(self):
        return 16 * 2 * self.weight.amax(self.quant_axis).numel() \
            + self.weight.numel() * self.n_bits \
            + 16 * 2 + self.bias.numel() * self.n_bits


class SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, inputs):
        return inputs * self.fc(inputs)


class Siren(nn.Module):
    def __init__(self, in_features, out_features, n_hidden_layers, hidden_dim,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        net = [SineLayer(in_features, hidden_dim, is_first=True,
                         omega_0=first_omega_0)]
        for i in range(n_hidden_layers):
            net.append(SineLayer(hidden_dim, hidden_dim,
                                 omega_0=hidden_omega_0))

        # output layer
        net.append(nn.Linear(hidden_dim, out_features))
        with torch.no_grad():
            net[-1].weight.uniform_(-np.sqrt(6/hidden_dim) / hidden_omega_0,
                                     np.sqrt(6/hidden_dim) / hidden_omega_0)
        net.append(nn.Tanh())

        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        return self.net(inputs)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(in_features)

    def init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6/in_features) / self.omega_0,
                     np.sqrt(6/in_features) / self.omega_0)

    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))

