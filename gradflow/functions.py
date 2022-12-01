from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any
from .tensor import Tensor
from .autograd import ReLUBackward

class Function(ABC):
  @abstractmethod
  def forward(self, *args: Any, **kwargs: Any) -> Any:
    pass

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self.forward(*args, **kwargs)

class ReLU(Function):
  def forward(self, x: Tensor):
    requires_grad = x.requires_grad
    is_leaf = not requires_grad

    out = Tensor(x.data * (x.data > 0), requires_grad, is_leaf)
    out.grad_fn = ReLUBackward([x], [x.grad_fn])
  
    return out

class Softmax(Function): # TODO: dim
  def forward(self, x: Tensor):
    # https://stackoverflow.com/a/34969389
    return x.exp() / x.exp().sum()