import mytorch.nn.functional as F
from mytorch.nn.module import Module
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ReLU(Module):
    """ReLU activation function

    Example:
    >>> relu = ReLU()
    >>> relu(input_tensor)
    <some output>

    We run this class like a function because of Module.__call__().

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ReLU forward pass.

        Args:
            x (Tensor): input before ReLU
        Returns:
            Tensor: input after ReLU
        """

        # Complete ReLU(Function) class in functional.py and call it appropriately here
        return F.ReLU.apply(x)

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.Tanh.apply(x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.Sigmoid.apply(x)
