from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional, Union
from .tensor import Tensor
from .autograd import ReLUBackward

import numpy as np

class Function(ABC):
  @abstractmethod
  def forward(self, *args: Any, **kwargs: Any) -> Tensor:
    pass

  def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
    return self.forward(*args, **kwargs)

class ParameterizedFunction(Function):
  @abstractmethod
  def parameters(self) -> list[Tensor]:
    pass

# Activation Functions:

class Sigmoid(Function):
  def forward(self, x: Tensor) -> Tensor:
    return 1 / (1 + (-x).exp())

class ReLU(Function):
  def forward(self, x: Tensor) -> Tensor:
    out = Tensor(x.data * (x.data > 0), x.requires_grad)
    out.grad_fn = ReLUBackward([x], [x.grad_fn])
  
    return out

class Softmax(Function):
  def __init__(self, dim: Optional[int] = None):
    self.dim = dim

  def forward(self, x: Tensor) -> Tensor:
    # shifts the values of x so that the highest number is 0, prevents numerical instability
    # see https://cs231n.github.io/linear-classify/#softmax-classifier
    x = x - np.max(x.data) 
    return x.exp() / x.exp().sum(self.dim)

class LogSoftmax(Function):
  def __init__(self, dim: Optional[int] = None):
    self.dim = dim
  
  def forward(self, x: Tensor) -> Tensor:
    x = x - np.max(x.data) 
    return x - x.exp().sum(self.dim).log().unsqueeze(-1)

class Tanh(Function):
  def forward(self, x: Tensor) -> Tensor:
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

# Loss functions

class MSELoss(Function):
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    return ((target - input) ** 2).mean()

class BCELoss(Function):
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    return (-(target * input.log() + (1 - target) * (1 - input).log())).mean() #.sum()

class NLLLoss(Function):
  # The negative log likelihood loss
  # https://en.wikipedia.org/wiki/Likelihood_function
  def __init__(self, indexed: bool = True):
    """
    `indexed=True` means the target is an array of indices of shape (bs,) e.g [0, 1, 5, 2, ...] 0 <= i < C
    `indexed=False` means the target is an array of distributions of shape (bs, C) e.g [[1,0,0,...],[0,1,0,...],...]
    """
    self.indexed = indexed

  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    if self.indexed:
      assert(input.shape != target.shape)
      # TODO: review
      target_mat = np.eye(input.shape[1])[target.data]
      target = Tensor(target_mat, requires_grad=False)

    return -(input * target).sum() / input.shape[0]

# Layers

class Dropout(Function):
  # Should be applied ONLY during training
  def __init__(self, p: float = 0.5):
    self.p = p

  def forward(self, x: Tensor) -> Tensor:
    mask = (np.random.randn(*x.shape) < self.p) / self.p
    return x * mask

class Linear(ParameterizedFunction):
  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    self._bias = bias

    # https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277 
    self.weight = Tensor(np.sqrt(2.0 / out_features) * np.random.randn(out_features, in_features), requires_grad=True)
    if bias:
      self.bias = Tensor(np.sqrt(2.0 / out_features) * np.random.randn(out_features), requires_grad=True)
  
  def parameters(self) -> list[Tensor]:
    params = [self.weight]
    params += [self.bias] if self._bias else []
    return params
    
  def forward(self, x: Tensor) -> Tensor:
    out = x @ self.weight.transpose()
    if self._bias:
      out = out + self.bias
    
    return out

class Conv2d(ParameterizedFunction):
  def __init__(self, in_features: int, out_features: int, kernel_size: Union[tuple[int, int], int], stride: int = 1, padding: str = "valid", bias: bool = True):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)*2
    
    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self._bias = bias

    self.weight = Tensor(np.sqrt(2.0 / out_features) * np.random.randn(out_features, in_features, *kernel_size), requires_grad=True)
    if bias:
      self.bias = Tensor(np.sqrt(2.0 / out_features) * np.random.randn(out_features), requires_grad=True)
  
  def parameters(self) -> list[Tensor]:
    params = [self.weight]
    params += [self.bias] if self._bias else []
    return params

  def forward(self, x: Tensor) -> Tensor:
    # TODO: padding
    bs, c, w, h = x.shape
    kw, kh = self.kernel_size[-2:]
    nb = x.data.dtype.itemsize

    new_w = (w - kw) // self.stride + 1
    new_h = (h - kh) // self.stride + 1

    # im2col
    patches = x.as_strided((bs, new_w, new_h, c, kw, kh),
                           (self.stride*w*h*nb, self.stride*h*nb, self.stride*nb, h*w*nb, w*nb, nb))
  
    col_mat = patches.reshape(bs * new_w * new_h, -1)

    out = col_mat @ self.weight.reshape(self.out_features, -1).transpose()
    out = out.transpose().reshape(bs, -1, new_h, new_w)

    if self._bias:
      out = out + self.bias.reshape(1, -1, 1, 1)

    return out

class MaxPool2D(Function):
  def __init__(self, kernel_size: Union[tuple[int, int], int], stride: int):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,) * 2

    self.kernel_size = kernel_size
    self.stride = stride
  
  def forward(self, x: Tensor) -> Tensor:
    bs, c, w, h = x.shape
    kw, kh = self.kernel_size
    nb = x.data.dtype.itemsize

    new_w = (w - kw) // self.stride + 1
    new_h = (h - kh) // self.stride + 1

    patches = x.as_strided((bs, c, new_w, new_h, kw, kh),
                           (c*w*h*nb, w*h*nb, self.stride*w*nb, self.stride*nb, w*nb, nb))
  
    return patches.max((-2, -1))
