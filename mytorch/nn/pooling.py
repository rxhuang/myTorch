import numpy as np

import mytorch.nn.functional as F
from mytorch.nn.module import Module
from mytorch.tensor import Tensor

class MaxPool2d(Module):
    """2D Max Pooling.
    
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

    Args:
        kernel_size (int): the size of the window to take a max over
        stride (int): the stride of the window. Default value is kernel_size.
    """
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
        Args:
            out (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        return F.MaxPool2d.apply(x, self.kernel_size, self.stride)
        
        
class AvgPool2d(Module):
    """2D Avg (Mean) Pooling.
    
    https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

    Args:
        kernel_size (int): the size of the window to take a mean over
        stride (int): the stride of the window. Default value is kernel_size.
    """
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
        Returns:
            out (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        return F.AvgPool2d.apply(x, self.kernel_size, self.stride)
