import unittest
from gradflow import *
import gradflow.functions as F
import numpy as np
import torch
import torch.nn as nn

class TestFunctions(unittest.TestCase):
  def __tensor_assert(self, tensor, grad, grad_fn, is_leaf, requires_grad):
    self.assertEqual(tensor.grad, grad)
    self.assertIsInstance(tensor.grad_fn, grad_fn)
    self.assertEqual(tensor.is_leaf, is_leaf)
    self.assertEqual(tensor.requires_grad, requires_grad)

  def __assert_numpy_equals(self, arr0, arr1, fp_err=1e-4):
    allclose = np.allclose(arr0, arr1, atol=fp_err)
    self.assertTrue(allclose)

  def test_sigmoid(self):
    sigmoid = F.Sigmoid()
    r = np.random.random((3, 2, 1, 3)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    b = sigmoid(a)
    tb = torch.sigmoid(ta)

    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    o = b.sum()
    to = tb.sum()

    o.backward()
    to.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_relu(self):
    relu = F.ReLU()
    r = np.random.random((3, 2, 1, 3)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    b = relu(a)
    tb = torch.relu(ta)

    self.__tensor_assert(b, None, F.ReLUBackward, False, True)
    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    o = b.sum()
    to = tb.sum()

    o.backward()
    to.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_softmax(self):
    r = np.random.random((3, 2, 1, 3)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for dim in [0, 1, 2, 3]:
      softmax = F.Softmax(dim)
      b = softmax(a)
      tb = torch.softmax(ta, dim)

      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      o = b.sum()
      to = tb.sum()

      o.backward()
      to.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)
  
  def test_logsoftmax(self):
    r = np.random.random((3, 2, 1, 3)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for dim in [0, 1, 2, 3]:
      logsoftmax = F.LogSoftmax(dim)
      b = logsoftmax(a)
      tb = torch.log_softmax(ta, dim)

      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      o = b.sum()
      to = tb.sum()

      o.backward()
      to.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_tanh(self):
    tanh = F.Tanh()
    r = np.random.random((3, 2, 1, 3)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    b = tanh(a)
    tb = torch.tanh(ta)

    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    o = b.sum()
    to = tb.sum()

    o.backward()
    to.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)
  
  # TODO
  def test_mse(self):
    pass

  # TODO
  def test_bce(self):
    pass

  # TODO
  def test_nll(self):
    pass

  def test_dropout(self):
    pass

  def test_linear(self):
    for bias in [True, False]:
      lin = F.Linear(10, 10, bias=bias)
      tlin = nn.Linear(10, 10, bias=bias)

      lin.weight.data = tlin.weight.detach().numpy()
      if bias:
        lin.bias.data = tlin.bias.detach().numpy()

      r = np.random.random((3, 10)).astype(np.float32)

      a = Tensor(r, requires_grad=True)
      ta = torch.tensor(r, requires_grad=True)

      b = lin(a)
      tb = tlin(ta)

      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      o = b.sum()
      to = tb.sum()

      o.backward()
      to.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_conv2d(self):
    pass
    # r = np.random.random((3, 3, 10, 10)).astype(np.float32)

    # a = Tensor(r, requires_grad=True)
    # ta = torch.tensor(r, requires_grad=True)

    # for bias in [False]:
    #   conv = F.Conv2d(3, 3, 3, bias=bias)
    #   tconv = nn.Conv2d(3, 3, 3, bias=bias)

    #   conv.weight.data = tconv.weight.detach().numpy()
    #   if bias:
    #     conv.bias.data = tconv.bias.detach().numpy()

    #   b = conv(a)
    #   tb = tconv(ta)

    #   self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    #   o = b.sum()
    #   to = tb.sum()

    #   o.backward()
    #   to.backward()

    #   self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_maxpool2d(self):
    pass
  
  
  # Future tests

  def test_avgpool2d(self):
    pass

  def test_batchnorm2d(self):
    pass

  def test_attention(self):
    pass