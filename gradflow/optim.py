from __future__ import annotations

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from .tensor import Tensor

import numpy as np

class Optimizer(ABC):
  def __init__(self, parameters: list[Tensor]):
    self.parameters = [param for param in parameters if param.requires_grad]
  
  def zero_grad(self) -> None:
    for param in self.parameters:
      param.grad = 0
  
  @abstractmethod
  def step(self) -> None:
    pass

# See: https://cs231n.github.io/neural-networks-3/#update

class SGD(Optimizer):
  def __init__(self, parameters: list[Tensor], lr: float, weight_decay: float, momentum: float, nesterov: bool):
    super().__init__(parameters)

    self.lr = lr
    self.weight_decay = weight_decay # L2 regularization parameter
    self.momentum = momentum
    self.nesterov = nesterov

    if self.momentum:
      self.v = [np.zeros_like(param.data) for param in self.parameters]
  
  def step(self):
    for i, param in enumerate(self.parameters):
      grad = param.grad
      if self.weight_decay:
        grad += self.weight_decay * param.data
      
      if self.momentum:
        if self.nesterov:
          v_prev = self.v[i]
          self.v[i] = self.momentum * self.v[i] - self.lr * grad
          param.data += -self.momentum * v_prev + (1 + self.momentum) * self.v[i]
        else:
          self.v[i] = self.momentum * self.v[i] - self.lr * grad
          param.data += self.v[i]
      else:
        param.data -= self.lr * grad

class Adagrad(Optimizer):
  def __init__(self, parameters: list[Tensor], lr: float, weight_decay: float, eps: float = 1e-8):
    super().__init__(parameters)

    self.lr = lr
    self.weight_decay = weight_decay # L2 regularization parameter
    self.eps = eps

    self.cache = [np.zeros_like(param.data) for param in self.parameters]

  def step(self) -> None:
    for i, param in enumerate(self.parameters):
      grad = param.grad
      if self.weight_decay:
        grad += self.weight_decay * param.data
      
      self.cache[i] += grad ** 2
      param.data += -self.lr * grad / (np.sqrt(self.cache[i]) + self.eps)
      
class RMSprop(Optimizer):
  def __init__(self, parameters: list[Tensor], lr: float, weight_decay: float, decay_rate: float, eps: float = 1e-8):
    super().__init__(parameters)

    self.lr = lr
    self.weight_decay = weight_decay # L2 regularization parameter
    self.decay_rate = decay_rate
    self.eps = eps

    self.cache = [np.zeros_like(param.data) for param in self.parameters]

  def step(self) -> None:
    for i, param in enumerate(self.parameters):
      grad = param.grad
      if self.weight_decay:
        grad += self.weight_decay * param.data
      
      self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * grad ** 2
      param.data += -self.lr * grad / (np.sqrt(self.cache[i]) + self.eps)

class Adam(Optimizer):
  def __init__(self, parameters: list[Tensor], lr: float, weight_decay: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    super().__init__(parameters)

    self.lr = lr
    self.weight_decay = weight_decay # L2 regularization parameter
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps

    self.v = [np.zeros_like(param.data) for param in self.parameters]
    self.m = [np.zeros_like(param.data) for param in self.parameters]

    self.t = 1

  def step(self) -> None:
    for i, param in enumerate(self.parameters):
      grad = param.grad
      if self.weight_decay:
        grad += self.weight_decay * param.data
      
      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
      mt = self.m[i] / (1 - self.beta1 ** self.t)
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
      vt = self.v[i] / (1 - self.beta2 ** self.t)
      param.data += -self.lr * mt / (np.sqrt(vt) + self.eps)

      self.t+=1
