""" Collection of helper functions and classes for the models package. """

import math
from typing import List

import torch
from torch import nn


class DebugLayer(nn.Module):
    """ Custom torch layer which can be used to print info or do stuff in between layers. """
    def __init__(self, description=''):
        super(DebugLayer, self).__init__()
        self.description = description

    def forward(self, x):
        # Do your print / debug stuff here
        print(f"{self.description} | Tensor shape: {x.shape}")
        return x


class ZeroPad3d(nn.ConstantPad3d):
    def __init__(self, padding):
        super().__init__(padding, 0)


def get_conv_dim_size(input_dim_size: int, kernel_size: int, padding: int, stride: int) -> int:
    """ Returns the output size of a feature map after applying convolution.

    Args:
        input_dim_size: Size of input (one dimension, e.g., x = 500)
        kernel_size: Size of conv kernel in that dimension
        padding: Amount of padding added (1 = a zero left and a zero right)
        stride: Stride in that dim

    Returns: Size of output feature map
    """
    return math.floor((input_dim_size-kernel_size+2*padding)/stride + 1)


def get_multiple_conv_dim_size(input_dim_size: int, kernel_sizes: List[int], paddings: List[int], strides: List[int]
                               ) -> int:
    """ Convenience function which applies get_conv_dim_size over multiple convolutions.

    Args:
        input_dim_size: Size of input (one dimension, e.g., x = 500)
        kernel_sizes: (list) Size of conv kernel in that dimension
        paddings: (list) Amount of padding added (1 = a zero left and a zero right)
        strides: (list) Stride in that dim

    Returns: Size of output feature map
    """
    dim_size = input_dim_size
    for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
        dim_size = get_conv_dim_size(dim_size, kernel_size, padding, stride)
    return dim_size


def get_paddings_indicator(actual_num, max_num, axis=0):
    """ Create boolean mask by actually number of a padded tensor. Used for feature encoders. """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num

    return paddings_indicator


# Run section ##########################################################################################################
########################################################################################################################


def main():
    pass

if __name__ == "__main__":
    main()
    