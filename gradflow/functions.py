from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional
from .tensor import Tensor
from .autograd import ReLUBackward

class Function(ABC):
  @abstractmethod
  def forward(self, *args: Any, **kwargs: Any) -> Tensor:
    pass

  def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
    return self.forward(*args, **kwargs)

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
    return x.exp() / x.exp().sum(self.dim)

class Tanh(Function):
  def forward(self, x: Tensor) -> Tensor:
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

# Loss functions

class MSELoss(Function):
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    return ((target - input) ** 2).mean()