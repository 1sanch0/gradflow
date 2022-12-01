from __future__ import annotations

from numpy.typing import ArrayLike
from typing import Union, Optional, Tuple
import numpy as np

from .autograd import *

# Based on https://www.youtube.com/watch?v=MswxJw-8PvE
class Tensor:
  def __init__(self, data: ArrayLike, requires_grad: bool = False, is_leaf: bool = True):
    self.data: np.ndarray = np.array(data, dtype=np.float32)
    self.grad: Optional[np.ndarray] = None
    self._grad_fn: AutogradFunction = Accumulate(self) if requires_grad else NoneFn()
    self.is_leaf: bool = is_leaf
    self._requires_grad: bool = requires_grad

  def astype(self, dtype: np.dtype) -> None:
    self.data.astype(dtype)
  
  @property
  def shape(self) -> Tuple[int]:
    return self.data.shape
  
  @property
  def grad_fn(self) -> AutogradFunction:
    return self._grad_fn

  @grad_fn.setter
  def grad_fn(self, fn: AutogradFunction) -> None:
    if self.requires_grad:
      self._grad_fn = fn

  @property
  def requires_grad(self) -> bool:
    return self._requires_grad  

  @requires_grad.setter
  def requires_grad(self, requires_grad: bool) -> None:
    if self.requires_grad == requires_grad:
      return None

    self._grad_fn = Accumulate(self) if requires_grad else NoneFn()
    self._requires_grad = requires_grad

  def __add__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs, False, False)
    requires_grad = self.requires_grad | rhs.requires_grad
    is_leaf = not requires_grad

    out = Tensor(self.data + rhs.data, requires_grad, is_leaf)
    out.grad_fn = AddBackward([self, rhs], [self.grad_fn, rhs.grad_fn])

    return out

  def __mul__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs, False, False)
    requires_grad = self.requires_grad | rhs.requires_grad
    is_leaf = not requires_grad

    out = Tensor(self.data * rhs.data, requires_grad, is_leaf)
    out.grad_fn = MulBackward([self, rhs], [self.grad_fn, rhs.grad_fn])

    return out

  def __matmul__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs, False, False)
    requires_grad = self.requires_grad | rhs.requires_grad
    is_leaf = not requires_grad

    out = Tensor(self.data @ rhs.data, requires_grad, is_leaf)
    out.grad_fn = MatmulBackward([self, rhs], [self.grad_fn, rhs.grad_fn])

    return out
  
  def __rmatmul__(self, lhs: Union[ArrayLike, Tensor]) -> Tensor:
    lhs = lhs if isinstance(lhs, Tensor) else Tensor(lhs, False, False)
    requires_grad = self.requires_grad | lhs.requires_grad
    is_leaf = not requires_grad

    out = Tensor(lhs.data @ self.data, requires_grad, is_leaf)
    out.grad_fn = MatmulBackward([lhs, self], [lhs.grad_fn, self.grad_fn])

    return out

  def transpose(self) -> Tensor:
    requires_grad = self.requires_grad
    is_leaf = not requires_grad

    out = Tensor(self.data.transpose(), requires_grad, is_leaf)
    out.grad_fn = TransposeBackward([self], [self.grad_fn])

    return out

  def sum(self) -> Tensor: # TODO: dim
    requires_grad = self.requires_grad
    is_leaf = not requires_grad

    out = Tensor(self.data.sum(), requires_grad, is_leaf)
    out.grad_fn = SumBackward([self], [self.grad_fn])

    return out

  def mean(self) -> Tensor: # TODO: dim
    requires_grad = self.requires_grad
    is_leaf = not requires_grad

    out = Tensor(self.data.mean(), requires_grad, is_leaf)
    out.grad_fn = MeanBackward(self.data.size, [self], [self.grad_fn])

    return out
  
  def exp(self) -> Tensor:
    requires_grad = self.requires_grad
    is_leaf = not requires_grad

    out = Tensor(np.exp(self.data), requires_grad, is_leaf)
    out.grad_fn = ExpBackward([self], [self.grad_fn])

    return out
  
  def __pow__(self, rhs: float) -> Tensor:
    assert(isinstance(rhs, float))
    requires_grad = self.requires_grad
    is_leaf = not requires_grad

    out = Tensor(np.power(self.data, rhs), requires_grad, is_leaf)
    out.grad_fn = PowBackward([self, rhs], [self.grad_fn])

    return out

  def __neg__(self) -> Tensor:
    return self * -1

  def __radd__(self, lhs: Union[ArrayLike, Tensor]) -> Tensor:
    return self + lhs

  def __sub__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    return self + (-rhs)

  def __rsub__(self, lhs: Union[ArrayLike, Tensor]) -> Tensor:
    return lhs + (-self)

  def __rmul__(self, lhs: Union[ArrayLike, Tensor]) -> Tensor:
    return self * lhs
  
  def __truediv__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    return self * rhs ** -1.0

  def __rtruediv__(self, lhs: Union[ArrayLike, Tensor]) -> Tensor:
    return lhs * self ** -1.0

  def backward(self) -> None:
    if (self.data.shape != ()):
      raise RuntimeError("grad can be implicitly created only for scalar outputs")
    self.grad_fn(1)

  def __repr__(self) -> str:
    grad_fn_name = self.grad_fn.__class__.__name__
    if grad_fn_name in ['Accumulate', 'NoneFn']:
      grad_fn_name = None

    out = f"Tensor({(self.data)}" 

    if grad_fn_name:
      out += f", grad_fn=<{grad_fn_name}>"

    out += ")"
    return out

