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
    x.data = x.data - np.max(x.data)#, axis=self.dim, keepdims=True)
    x_e = x.exp()
    return x_e / x_e.sum(self.dim, keepdims=True)

class LogSoftmax(Function):
  def __init__(self, dim: Optional[int] = None):
    self.dim = dim
  
  def forward(self, x: Tensor) -> Tensor:
    x.data = x.data - np.max(x.data)
    return x - x.exp().sum(self.dim, keepdims=True).log()#.unsqueeze(-1)

class Tanh(Function):
  def forward(self, x: Tensor) -> Tensor:
    x_exp = x.exp()
    x_neg_exp = (-x).exp()
    return (x_exp - x_neg_exp) / (x_exp + x_neg_exp)

# Loss functions

class MSELoss(Function):
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    return ((target - input) ** 2).mean()

class BCELoss(Function):
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    return (-(target * input.log() + (1 - target) * (1 - input).log())).mean()

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
  def __init__(self, p: float = 0.5):
    self.p = p

  # TODO: train global variable
  def forward(self, x: Tensor) -> Tensor:
    mask = (np.random.randn(*x.shape) < self.p) / (1-self.p)
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

def im2col(x: Tensor, kernel_size: tuple[int, int], strides: tuple[int, int]) -> Tensor:
  bs, c, h, w = x.shape
  kh, kw = kernel_size
  sh, sw = strides

  out_h = (h - kh) // sh + 1
  out_w = (w - kw) // sw + 1

  patches = x.as_strided((bs, c, out_h, out_w, kh, kw),
                         (c*h*w, h*w, sh*w, sw, w, 1)) \
             .transpose((0, 1, 4, 5, 2, 3))
  
  return patches.reshape(bs, c * kh * kw, out_h * out_w)

class Conv2d(ParameterizedFunction):
  def __init__(self, in_features: int, out_features: int, kernel_size: Union[tuple[int, int], int], stride: Union[tuple[int, int], int] = 1, padding: str = "valid", bias: bool = True):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)*2

    if isinstance(stride, int):
      stride = (stride,)*2

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
    # See examples/how_to_conv.ipynb for more info about this approach
    bs, c, h, w = x.shape
    kh, kw = self.kernel_size
    sh, sw = self.stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    if c != self.in_features:
      raise ValueError(f"Expected {self.in_features} channels, but got {c}")

    x_col = im2col(x, self.kernel_size, self.stride)
    x_col.data = np.ascontiguousarray(x_col.data)

    weight_col = self.weight.reshape(self.out_features, -1)

    out = (weight_col @ x_col).reshape(bs, self.out_features, out_h, out_w)

    if self._bias:
      out = out + self.bias.reshape(1, -1, 1, 1)

    return out

class MaxPool2d(Function):
  def __init__(self, kernel_size: Union[tuple[int, int], int], stride: Union[tuple[int, int], int] = 1):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,) * 2
    
    if isinstance(stride, int):
      stride = (stride,) * 2

    self.kernel_size = kernel_size
    self.stride = stride
  
  def forward(self, x: Tensor) -> Tensor:
    # TODO: padding
    bs, c, h, w = x.shape
    kh, kw = self.kernel_size
    sh, sw = self.stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    patches = x.as_strided((bs, c, out_h, out_w, kh, kw),
                           (c*h*w, h*w, sh*w, sw, w, 1))

    return patches.max((-2, -1))
  
class BatchNorm2d(ParameterizedFunction):
  def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats

    self.weight = Tensor(np.ones(num_features), requires_grad=True)
    self.bias = Tensor(np.zeros(num_features), requires_grad=True)

    self.running_mean = Tensor(np.zeros(num_features), requires_grad=False)
    self.running_var = Tensor(np.ones(num_features), requires_grad=False)
  
  def parameters(self) -> list[Tensor]:
    return [self.weight, self.bias]

  # TODO: Global variable train
  def forward(self, x: Tensor, train: bool = True) -> Tensor:
    if train:
      mean = x.mean((0, 2, 3))
      var = x.var((0, 2, 3), correction=0)

      if self.track_running_stats:
        self.running_mean = (1 - self.momentum) * self.running_mean + (self.momentum) * mean
        self.running_var = (1 - self.momentum) * self.running_var + (self.momentum) * x.var((0, 2, 3), correction=1)
    else:
      mean = self.running_mean
      var = self.running_var

    x_hat = (x - mean.reshape(1, -1, 1, 1)) / (var.reshape(1, -1, 1, 1) + self.eps) ** 0.5
    if self.affine:
      return self.weight.reshape(1, -1, 1, 1) * x_hat + self.bias.reshape(1, -1, 1, 1)
    return x_hat
