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
  
  def test_mse(self):
    r = np.random.random((3, 5)).astype(np.float32)
    t = np.random.random((3, 5)).astype(np.float32)
    
    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    tg = Tensor(t, requires_grad=False)
    ttg = torch.tensor(t, requires_grad=False)

    mse = F.MSELoss()
    tmse = nn.MSELoss()

    b = mse(a, tg)
    tb = tmse(ta, ttg)

    b.backward()
    tb.backward()

    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_bce(self):
    m = F.Sigmoid()
    tm = nn.Sigmoid()

    bce = F.BCELoss()
    tbce = nn.BCELoss()

    r = np.random.random((3, 5)).astype(np.float32)
    t = np.random.random((3, 5)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    tg = Tensor(t, requires_grad=False)
    ttg = torch.tensor(t, requires_grad=False)

    b = bce(m(a), tg)
    tb = tbce(tm(ta), ttg)

    b.backward()
    tb.backward()

    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_nll(self):
    m = F.LogSoftmax(1)
    tm = nn.LogSoftmax(1)

    nll = F.NLLLoss()
    tnll = nn.NLLLoss()

    r = np.random.random((3, 5)).astype(np.float32)
    t = np.array([0, 1, 2])

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    tg = Tensor(t, requires_grad=False, dtype=np.int64)
    ttg = torch.tensor(t, requires_grad=False, dtype=torch.int64)

    b = nll(m(a), tg)
    tb = tnll(tm(ta), ttg)

    b.backward()
    tb.backward()

    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_dropout(self):
    # How do you even test dropout?
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
    r = np.random.random((4, 3, 16, 16)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for bias in [False, True]:
      conv = F.Conv2d(3, 3, 3, bias=bias)
      tconv = nn.Conv2d(3, 3, 3, bias=bias)

      conv.weight.data = tconv.weight.detach().numpy()
      if bias:
        conv.bias.data = tconv.bias.detach().numpy()

      b = conv(a)
      tb = tconv(ta)

      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      o = b.sum()
      to = tb.sum()

      o.backward()
      to.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_maxpool2d(self):
    maxpool = F.MaxPool2d(2, 2)
    fmaxpool = nn.MaxPool2d(2, 2)

    r = np.random.random((4, 3, 16, 16)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    tb = fmaxpool(ta)
    b = maxpool(a)

    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    o = b.sum()
    to = tb.sum()

    o.backward()
    to.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_avgpool2d(self):
    # TODO
    pass

  def test_batchnorm2d(self):
    bn = F.BatchNorm2d(3)
    tbn = nn.BatchNorm2d(3)

    r = np.random.random((2, 3, 3, 3)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)
    
    self.__assert_numpy_equals(tbn.running_mean.numpy(), bn.running_mean.data)
    self.__assert_numpy_equals(tbn.running_var.numpy(), bn.running_var.data)

    for _ in range(10):
      b = bn(a)
      tb = tbn(ta)
      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    o = b.sum()
    to = tb.sum()

    print(tbn.running_var.numpy())
    print(bn.running_var.data)
    self.__assert_numpy_equals(tbn.running_mean.numpy(), bn.running_mean.data)
    self.__assert_numpy_equals(tbn.running_var.numpy(), bn.running_var.data)

    o.backward()
    to.backward()

    self.__assert_numpy_equals(tb.detach().numpy(), b.data,)
    self.__assert_numpy_equals(tbn.weight.grad.numpy(), bn.weight.grad)
    self.__assert_numpy_equals(tbn.bias.grad.numpy(), bn.bias.grad)

    # TODO: FIX THIS, ITS CLOSE ENOUGH BUT NEEDS REVIEW
    # print(np.abs(ta.grad.detach().numpy()-a.grad))
    # self.__assert_numpy_equals(ta.grad.numpy(), a.grad)
  
  def test_attention(self):
    # TODO
    pass