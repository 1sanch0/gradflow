from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional
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
    requires_grad = x.requires_grad
    is_leaf = not requires_grad

    out = Tensor(x.data * (x.data > 0), requires_grad, is_leaf)
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
      print("input:", input.shape)
      print("Target mat:", target_mat.shape)

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
