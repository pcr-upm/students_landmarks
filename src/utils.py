#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Alejandro Cobo'
__email__ = 'alejandro.cobo@upm.es'

import torch
import torch.nn.functional as F
from math import floor


def get_landmarks_local_softmax(output, temperature, window, device):
    """
    Window size must be 3 (3x3), 5 (5x5) or 7 (7x7)
    """
    batch_size, nb_channels, h, w = output.size()
    output = output * temperature
    # Define convolution
    conv = torch.nn.Conv2d(nb_channels, nb_channels, window, groups=nb_channels, padding=floor(window/2), bias=False).to(device)
    kernel = torch.ones(window, window).view(1, 1, window, window).repeat(nb_channels, 1, 1, 1).to(device)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(kernel)
    # Get mask with max values
    mask = (output == torch.amax(output, dim=(2, 3), keepdim=True)).to(device)
    masked_output = output.clone().to(device)
    masked_output[torch.logical_not(mask)] = 0
    windowed_masked_output = conv(masked_output).to(device)
    masked_output[windowed_masked_output > 0] = output[windowed_masked_output > 0]
    # To perform local softmax we have to cancel out all other terms (e^-inf)
    masked_output[windowed_masked_output == 0] = -float('inf')
    # Global soft-argmax on windowed masked output
    arange = torch.arange(0, masked_output.size(3)).to(device)
    probs = F.softmax(masked_output.reshape(*masked_output.shape[:-2], -1), dim=-1).view_as(masked_output).to(device)
    # Compute coordinates
    i = torch.sum(torch.sum(probs, dim=(-2)) * arange, dim=-1, keepdim=True).to(device)
    j = torch.sum(torch.sum(probs, dim=(-1)) * arange, dim=-1, keepdim=True).to(device)
    return torch.cat([i, j], dim=-1).to(device)

