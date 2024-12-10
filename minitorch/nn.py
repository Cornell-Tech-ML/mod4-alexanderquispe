from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling"""
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    
    new_height = height // kh
    new_width = width // kw
    
    input = input.contiguous()
    # Reshape to batch x channel x new_height x kernel_height x new_width x kernel_width
    out = input.view(batch, channel, new_height, kh, new_width, kw)
    # Permute to get desired order
    out = out.permute(0, 1, 2, 4, 3, 5)
    # Reshape to combine last two dimensions
    out = out.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return out, new_height, new_width

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    batch, channel, height, width = input.shape
    kh, kw = kernel
    
    # Get tiled input
    tiled, new_height, new_width = tile(input, kernel)
    
    # Calculate mean over the last dimension (kernel area)
    kernel_size = kh * kw
    out = tiled.sum(dim=-1) / kernel_size
    
    # Ensure output has correct shape
    return out.view(batch, channel, new_height, new_width)

def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    out = zeros(input.shape)
    max_indices = input.max(dim)[1]
    
    # Create a 1-hot encoding
    for i in range(len(max_indices)):
        out[i, max_indices[i]] = 1.0
    return out

class Max(Function):
    """Max function"""
    @staticmethod
    def forward(ctx: Context, input: Tensor) -> Tensor:
        ctx.save_for_backward(input)
        return input.max(0)[0]

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        input = ctx.saved_values
        return grad_output * (input == input.max(0)[0].broadcast_to(input.shape))

def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input)

def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor"""
    # Subtract max for numerical stability (log-sum-exp trick)
    max_vals = input.max(dim)[0]
    exp_vals = (input - max_vals.broadcast_to(input.shape)).exp()
    sum_vals = exp_vals.sum(dim)
    return exp_vals / sum_vals.broadcast_to(input.shape)

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor"""
    # Use the log-sum-exp trick for numerical stability
    max_vals = input.max(dim)[0]
    shifted = input - max_vals.broadcast_to(input.shape)
    exp_vals = shifted.exp()
    sum_vals = exp_vals.sum(dim)
    return shifted - log(sum_vals).broadcast_to(input.shape)

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""
    batch, channel, height, width = input.shape
    
    # Get tiled input
    tiled, new_height, new_width = tile(input, kernel)
    
    # Max over the last dimension (kernel_size)
    return tiled.max(dim=-1)[0]

def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise"""
    if ignore:
        return input
    
    # Generate random mask
    rand = rand_like(input)
    mask = rand > rate
    
    # Scale output to maintain expected value
    scale = 1.0 / (1.0 - rate)
    return mask.float() * input * scale