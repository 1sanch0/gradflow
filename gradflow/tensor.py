from __future__ import annotations

from numpy.typing import ArrayLike
from typing import Union, Optional, Tuple
import numpy as np

from .autograd import *

# Based on https://www.youtube.com/watch?v=MswxJw-8PvE
class Tensor:
  def __init__(self, data: ArrayLike, requires_grad: bool = False, dtype: np.dtype = np.float32):
    self.data: np.ndarray = np.array(data, dtype=dtype)
    self.grad: Optional[np.ndarray] = None
    self._grad_fn: AutogradFunction = Accumulate(self) if requires_grad else NoneFn()
    self._requires_grad: bool = requires_grad

  # def astype(self, dtype: np.dtype) -> None:
  #   self.data.astype(dtype)
  
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
  def is_leaf(self) -> bool:
    if (not self.requires_grad): return True
    if (isinstance(self.grad_fn, Accumulate) or isinstance(self.grad_fn, NoneFn)):
      return True
    return False

  @property
  def requires_grad(self) -> bool:
    return self._requires_grad  

  @requires_grad.setter
  def requires_grad(self, requires_grad: bool) -> None:
    if (not self.is_leaf):
      raise RuntimeError("you can only change requires_grad of leaf variables.")
    
    if self.requires_grad == requires_grad:
      return None

    self._grad_fn = Accumulate(self) if requires_grad else NoneFn()
    self._requires_grad = requires_grad
  
  def detach(self) -> Tensor:
    return Tensor(self.data, requires_grad=False)

  def __add__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs, False)

    out = Tensor(self.data + rhs.data, self.requires_grad | rhs.requires_grad)
    out.grad_fn = AddBackward([self, rhs], [self.grad_fn, rhs.grad_fn])

    return out

  def __mul__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs, False)

    out = Tensor(self.data * rhs.data, self.requires_grad | rhs.requires_grad)
    out.grad_fn = MulBackward([self, rhs], [self.grad_fn, rhs.grad_fn])

    return out

  def __matmul__(self, rhs: Union[ArrayLike, Tensor]) -> Tensor:
    rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs, False)

    out = Tensor(self.data @ rhs.data, self.requires_grad | rhs.requires_grad)
    out.grad_fn = MatmulBackward([self, rhs], [self.grad_fn, rhs.grad_fn])

    return out
  
  def __rmatmul__(self, lhs: Union[ArrayLike, Tensor]) -> Tensor:
    lhs = lhs if isinstance(lhs, Tensor) else Tensor(lhs, False)

    out = Tensor(lhs.data @ self.data, self.requires_grad | lhs.requires_grad)
    out.grad_fn = MatmulBackward([lhs, self], [lhs.grad_fn, self.grad_fn])

    return out

  def transpose(self, dim: Optional[tuple[int, ...]] = None) -> Tensor:
    out = Tensor(self.data.transpose(dim), self.requires_grad)
    out.grad_fn = TransposeBackward([dim], [self.grad_fn])

    return out

  def max(self, dim: Optional[tuple[int, ...]] = None) -> Tensor:
    # self.data = self.data.astype(np.float64)
    out = Tensor(self.data.max(dim), self.requires_grad)
    out.grad_fn = MaxBackward([self.data, out.data, dim], [self.grad_fn])

    return out

  def sum(self, dim: Optional[int] = None, keepdims: bool = False) -> Tensor:
    out = Tensor(self.data.sum(dim, keepdims=keepdims), self.requires_grad)
    out.grad_fn = SumBackward([self], [self.grad_fn], dim)

    return out

  def mean(self, dim: Optional[int] = None) -> Tensor:
    return self.sum(dim) / self.data.size
  
  def exp(self) -> Tensor:
    out = Tensor(np.exp(self.data), self.requires_grad)
    out.grad_fn = ExpBackward([self], [self.grad_fn])

    return out

  def log(self) -> Tensor:
    ''' Natural logarithm (base e) '''
    out = Tensor(np.log(self.data), self.requires_grad)
    out.grad_fn = LogBackward([self], [self.grad_fn])

    return out
  
  def __pow__(self, rhs: float) -> Tensor:
    assert(isinstance(rhs, (int, float)))

    out = Tensor(np.power(self.data, rhs), self.requires_grad)
    out.grad_fn = PowBackward([self, rhs], [self.grad_fn])

    return out
  
  def squeeze(self, dim: int) -> Tensor:
    """Remove axes of length one from x."""

    out = Tensor(np.squeeze(self.data, dim), self.requires_grad)
    out.grad_fn = SqueezeBackward([self], [self.grad_fn], dim)

    return out

  def unsqueeze(self, dim: int) -> Tensor:
    """Expand the shape of an array."""

    out = Tensor(np.expand_dims(self.data, dim), self.requires_grad)
    out.grad_fn = UnsqueezeBackward([self], [self.grad_fn], dim)

    return out
  
  def reshape(self, *shape: int) -> Tensor:
    out = Tensor(self.data.reshape(shape), self.requires_grad)
    out.grad_fn = ReshapeBackward([self.data.shape], [self.grad_fn])

    return out
  
  def as_strided(self, shape: tuple[int, ...], strides: tuple[int, ...]) -> Tensor:
    """ This function has to be used with extreme care. """
    nb = self.data.dtype.itemsize

    out = Tensor(
      np.ascontiguousarray(np.lib.stride_tricks.as_strided(self.data, shape, nb*np.array(strides))) # TODO: benchmark
    , self.requires_grad)
    out.grad_fn = AsStridedBackward([self.data.shape, self.data.strides, shape, strides], [self.grad_fn])

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
    self.grad_fn(np.ones_like(self.data))

  def __str__(self) -> str:
    grad_fn_name = self.grad_fn.__class__.__name__
    if grad_fn_name in ['Accumulate', 'NoneFn']:
      grad_fn_name = None

    out = f"Tensor({(self.data)}" 

    if grad_fn_name:
      out += f", grad_fn=<{grad_fn_name}>"

    out = out.replace("\n", "\n" + " " * len("Tensor("))
    out += ")"
    return out
  
  def __repr__(self) -> str:
    return str(self)
