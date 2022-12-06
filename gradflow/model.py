from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any
from .functions import ParameterizedFunction
from .tensor import Tensor

class Model(ABC):
  def __init__(self):
    self.params = [] 

  def __setattr__(self, name: str, value: Any) -> None:
    if hasattr(self, name):
      raise AttributeError(f"Attribute \"{name}\" already exists")
    
    super().__setattr__(name, value)
    if isinstance(value, (Tensor, Model, ParameterizedFunction)):
      self.params.extend(value.parameters())

  def parameters(self) -> list[Tensor]:
    return self.params

  @abstractmethod
  def forward(self, *args: Any, **kwargs: Any) -> Tensor:
    pass

  def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
    return self.forward(*args, **kwargs)
