from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .tensor import Tensor

import numpy as np

class AutogradFunction(ABC):
  @abstractmethod
  def backward(self, grad: np.ndarray) -> None:
    pass

  def __call__(self, grad: np.ndarray) -> None:
    self.backward(grad)

class BackwardFunction(AutogradFunction):
  def __init__(self, ctx: list[Tensor], next_functions: list[BackwardFunction]): 
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

class AddBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](grad)
    self.next_functions[1](grad)

class MulBackward(BackwardFunction):
  # Binary op: a (op) b
  # ctx[0] = a, ctx[1] = b
  # next_function[0] = a.grad_fn
  # next_function[1] = b.grad_fn
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](self.ctx[1].data * grad)
    self.next_functions[1](self.ctx[0].data * grad)

class MatmulBackward(BackwardFunction):
  # Binary op: a (op) b
  # ctx[0] = a, ctx[1] = b
  # next_function[0] = a.grad_fn
  # next_function[1] = b.grad_fn
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](grad @ self.ctx[1].data.T)
    self.next_functions[1](self.ctx[0].data.T @ grad)

class TransposeBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](grad.transpose())

class SumBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](np.ones_like(self.ctx[0].data) * grad)

class MeanBackward(BackwardFunction):
  def __init__(self, n: int, ctx: list[Tensor], next_functions: list[BackwardFunction]):
    super().__init__(ctx, next_functions)
    self.n = n

  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](np.ones_like(self.ctx[0].data) * grad / self.n)

class ExpBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](np.exp(self.ctx[0].data) * grad)

class PowBackward(BackwardFunction):
  # In this case, ctx[1] = rhs
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0](self.ctx[1] * np.power(self.ctx[0].data, self.ctx[1] - 1) * grad)

class ReLUBackward(BackwardFunction):
  def backward(self, grad: np.ndarray) -> None:
    self.next_functions[0]((self.ctx[0].data > 0) * grad)
