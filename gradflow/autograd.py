from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
  from .tensor import Tensor

import numpy as np

# Binary ops: a (op) b
# ctx[0] = a, ctx[1] = b
# next_function[0] = a.grad_fn
# next_function[1] = b.grad_fn

# Unary ops: a.(op)
# ctx[0] = a
# next_function[0] = a.grad_fn

class AutogradFunction(ABC):
  @abstractmethod
  def backward(self, grad: np.ndarray) -> None:
    pass

  def __call__(self, grad: np.ndarray) -> None:
    self.backward(grad)

class BackwardFunction(AutogradFunction):
  def __init__(self, ctx: list[Tensor], next_functions: list[AutogradFunction]): 
    self.ctx = ctx
    self.next_functions = next_functions

class Accumulate(AutogradFunction):
  def __init__(self, tensor: Tensor):
    self.tensor: Tensor = tensor
  
  def backward(self, grad: np.ndarray) -> None:
    if (not isinstance(self.tensor.grad, np.ndarray)):
      if (self.tensor.grad == None):
        self.tensor.grad = 0

    self.tensor.grad += grad

class NoneFn(AutogradFunction):
  def backward(self, grad: np.ndarray) -> None:
    pass

# General Broadcasting Rules (https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
# It starts with the trailing (i.e. rightmost) dimension and works its way left.
# Two dimensions are compatible when
# + they are equal, or
# + one of them is 1.
def unbroadcast(x: np.ndarray, shape: tuple[int]) -> np.ndarray:
  if (x.shape == shape):
    return x

  if (shape == (1,) or shape == ()):
    return x.sum()

  # Missing dimensions are assumed to have size one.
  shape_ = (1,) * (len(x.shape) - len(shape)) + shape

  expanded_dims = np.where(np.array(x.shape) != np.array(shape_))[0]

  # print("X shape:", x.shape)
  # print("Shape:", shape)
  # print("!=", np.array(shape) != np.array(x.shape))
  # print("np.where:", np.where(np.array(x.shape) != np.array(shape)))
  # print("Expanded dims:", expanded_dims)

  return x.sum(tuple(expanded_dims)).reshape(shape)

class AddBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](unbroadcast(grad, self.ctx[0].shape))
    self.next_functions[1](unbroadcast(grad, self.ctx[1].shape))

class MulBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](unbroadcast(self.ctx[1].data * grad, self.ctx[0].shape))
    self.next_functions[1](unbroadcast(self.ctx[0].data * grad, self.ctx[1].shape))

class MatmulBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](grad @ self.ctx[1].data.T)
    self.next_functions[1](self.ctx[0].data.T @ grad)

class TransposeBackward(BackwardFunction):
  # ctx[0] = dim
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](grad.transpose(self.ctx[0]))

class MaxBackward(BackwardFunction):
  # ctx[0] = patches
  # ctx[1] = maxes
  # ctx[2] = dim
  def backward(self, grad: np.ndarray) -> None:
    indices = np.where(self.ctx[0] == np.expand_dims(self.ctx[1], self.ctx[2]))
    new_grad = np.zeros_like(self.ctx[0])
    new_grad[indices] = grad.ravel()

    self.next_functions[0](new_grad)

class SumBackward(BackwardFunction):
  def __init__(self, ctx: list[Tensor], next_functions: list[AutogradFunction], dim: Optional[int] = None):
    super().__init__(ctx, next_functions)
    self.dim = dim

  def backward(self, grad: np.ndarray) -> None:
    # if self.dim:
    #   grad = np.expand_dims(grad, self.dim)
    self.next_functions[0](np.ones_like(self.ctx[0].data) * grad)

class ExpBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](np.exp(self.ctx[0].data) * grad)

class LogBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](grad / self.ctx[0].data)

class PowBackward(BackwardFunction):
  # In this case, ctx[1] = rhs
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](self.ctx[1] * np.power(self.ctx[0].data, self.ctx[1] - 1) * grad)

class ReLUBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0]((self.ctx[0].data > 0) * grad)

class UnsqueezeBackward(BackwardFunction):
  def __init__(self, ctx: list[Tensor], next_functions: list[AutogradFunction], dim: int):
    super().__init__(ctx, next_functions)
    self.dim = dim

  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](np.squeeze(grad, self.dim))

class SqueezeBackward(BackwardFunction):
  def __init__(self, ctx: list[Tensor], next_functions: list[AutogradFunction], dim: int):
    super().__init__(ctx, next_functions)
    self.dim = dim

  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](np.expand_dims(grad, self.dim))

class ReshapeBackward(BackwardFunction):
  # ctx[0] = old shape
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](grad.reshape(self.ctx[0]))

class AsStridedBackward(BackwardFunction):
  #ctx[0] = old shape
  #ctx[1] = old strides
  #ctx[2] = input shape
  #ctx[3] = input strides
  def backward(self, grad: np.ndarray) -> None:
    new_grad = np.zeros(self.ctx[0]).flatten()
    idx = np.arange(np.prod(self.ctx[0]))
    indices = np.lib.stride_tricks.as_strided(
      idx, self.ctx[2], np.array(self.ctx[3]) * idx.dtype.itemsize
    )
    
    np.add.at(new_grad, indices.flatten(), grad.flatten())

    self.next_functions[0](new_grad.reshape(self.ctx[0]))
