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
    a: np.array = self.ctx[0].data
    b: np.array = self.ctx[1].data

    # print(f"\nMatmulBackward: {a.shape} @ {b.shape} with grad: {grad.shape}")

    a_prepended, b_appended = False, False
    a_broadcasted_dims, b_broadcasted_dims = None, None
    if not (a.shape == b.shape and a.ndim > 1): # Broadcasting https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
      if (a.ndim == 1):
        a_prepended = True
        a = a.reshape(1, -1)
        # print("a_dims == 1, a prepended with 1")
      
      if (b.ndim == 1):
        b_appended = True
        b = b.reshape(-1, 1)
        # print("b_dims == 1, b appended with 1")
      
      if (a.ndim > b.ndim):
        n = a.ndim - b.ndim
        b_broadcasted_dims = tuple(range(n))
        b = b.reshape((1,)*n + b.shape)
        # print(f"a_dims > b_dims, b broadcasted {b_broadcasted_dims}")
      
      if (b.ndim > a.ndim):
        n = b.ndim - a.ndim
        a_broadcasted_dims = tuple(range(n))
        a = a.reshape((1,)*n + a.shape)
        # print(f"b_dims > a_dims, a broadcasted {a_broadcasted_dims}")

    # print(f"After broadcast: {a.shape} @ {b.shape} with grad: {grad.shape}")

    a_T_dims, b_T_dims = list(range(a.ndim)), list(range(b.ndim))
    a_T_dims[-2], a_T_dims[-1] = a_T_dims[-1], a_T_dims[-2]
    b_T_dims[-2], b_T_dims[-1] = b_T_dims[-1], b_T_dims[-2]

    a_T, b_T = a.transpose(a_T_dims), b.transpose(b_T_dims)

    shape = (np.zeros_like(a) @ np.zeros_like(b)).shape # TODO: Find a better way to get the shape of the result
    grad = grad.reshape(shape)
    # print(f"R: {(a@b).shape}")
    # print(f"Grad_0: {grad.shape} @ {b_T.shape}")
    # print(f"Grad_1: {a_T.shape} @ {grad.shape}")

    grad_0 = grad @ b_T
    grad_1 = a_T @ grad

    # print(f"Grad_0: {grad_0.shape}")
    # print(f"Grad_1: {grad_1.shape}")
    # print(f"ctx[0]: {self.ctx[0].shape}")
    # print(f"ctx[1]: {self.ctx[1].shape}")

    if (a_prepended):
      grad_0 = grad_0.sum(-2)
    
    if (b_appended):
      grad_1 = grad_1.sum(-1)

    if (a_broadcasted_dims):
      grad_0 = grad_0.sum(a_broadcasted_dims)

    if (b_broadcasted_dims):
      grad_1 = grad_1.sum(b_broadcasted_dims)

    # print(f"ubc Grad_0: {grad_0.shape}")
    # print(f"ubc Grad_1: {grad_1.shape}")

    self.next_functions[0](grad_0)
    self.next_functions[1](grad_1)

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
