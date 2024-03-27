import unittest
from gradflow import *
import numpy as np
import torch

class TestTensor(unittest.TestCase):
  def __tensor_assert(self, tensor, grad, grad_fn, is_leaf, requires_grad):
    self.assertEqual(tensor.grad, grad)
    self.assertIsInstance(tensor.grad_fn, grad_fn)
    self.assertEqual(tensor.is_leaf, is_leaf)
    self.assertEqual(tensor.requires_grad, requires_grad)

  def __assert_numpy_equals(self, arr0, arr1, fp_err=1e-4):
    self.assertTrue(np.allclose(arr0, arr1, atol=fp_err))

  def test_detach(self):
    a = Tensor(2.0, requires_grad=True)
    self.__tensor_assert(a, None, Accumulate, True, True)
    b = a.detach()
    self.__tensor_assert(b, None, NoneFn, True, False)
  
  # Test the *few* tensor primitives
  
  def test_sum(self):
    r = np.random.random((3, 2, 1, 3)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for dim in [None, 0, 1, 2, 3]:
      o = a.sum(dim)
      to = ta.sum(dim)
      self.__tensor_assert(o, None, SumBackward, False, True)
      self.__assert_numpy_equals(to.detach().numpy(), o.data)

    o = a.sum()
    to = ta.sum()

    o.backward()
    to.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)    

  # mean should work fine if sum does

  def test_add(self):
    r1 = np.random.random((10, 4, 1, 9, 4)).astype(np.float32)
    r2 = np.random.random((10, 4, 1, 9, 4)).astype(np.float32)

    a = Tensor(r1, requires_grad=True)
    b = Tensor(r2, requires_grad=True)

    ta = torch.tensor(r1, requires_grad=True)
    tb = torch.tensor(r2, requires_grad=True)

    c = a + b
    tc = ta + tb

    self.__tensor_assert(c, None, AddBackward, False, True)
    self.__assert_numpy_equals(tc.detach().numpy(), c.data)

    c = c.sum()
    tc = tc.sum()

    c.backward()
    tc.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)
    self.__assert_numpy_equals(tb.grad.numpy(), b.grad)

    # TODO test broadcast

  def test_mul(self):
    r1 = np.random.random((10, 4, 4, 1, 9, 4)).astype(np.float32)
    r2 = np.random.random((10, 4, 4, 1, 9, 4)).astype(np.float32)

    a = Tensor(r1, requires_grad=True)
    b = Tensor(r2, requires_grad=True)

    ta = torch.tensor(r1, requires_grad=True)
    tb = torch.tensor(r2, requires_grad=True)

    c = a * b
    tc = ta * tb

    self.__tensor_assert(c, None, MulBackward, False, True)
    self.__assert_numpy_equals(tc.detach().numpy(), c.data)

    d = c.sum()
    td = tc.sum()

    d.backward()
    td.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)
    self.__assert_numpy_equals(tb.grad.numpy(), b.grad)

  def test_matmul(self):
    dims = [((4, 9), (9, 4)),

            ((1,), (1,)),
            ((6,), (6,)),

            ((6,), (6, 2)),
            ((3, 4), (4,)),


            ((3, 4, 7), (3, 7, 1)),
            ((3, 4, 7), (7, 1)),
            ((3, 4, 7), (7,)),
            ((4, 7), (3, 7, 1)),
            ((7), (3, 7, 1)),

            ((9, 4, 7, 4), (9, 4, 4, 3)),
            ((9, 4, 7, 4), (4, 4, 3)),
            ((9, 4, 7, 4), (4, 3)),
            ((9, 4, 7, 4), (4)),

            ((9, 4, 7, 4), (9, 4, 4, 3)),
            ((4, 7, 4), (9, 4, 4, 3)),
            ((7, 4), (9, 4, 4, 3)),
            ((4), (9, 4, 4, 3)),
            ]

    for d1, d2 in dims:
      print(f"Testing {d1} @ {d2}")
      r1 = np.random.random(d1).astype(np.float32)
      r2 = np.random.random(d2).astype(np.float32)

      a = Tensor(r1, requires_grad=True)
      b = Tensor(r2, requires_grad=True)

      ta = torch.tensor(r1, requires_grad=True)
      tb = torch.tensor(r2, requires_grad=True)

      tc = ta @ tb
      c = a @ b

      self.__tensor_assert(c, None, MatmulBackward, False, True)
      self.__assert_numpy_equals(tc.detach().numpy(), c.data)

      d = c.mean()
      td = tc.mean()

      td.backward()
      d.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)
      self.__assert_numpy_equals(tb.grad.numpy(), b.grad)
  
  def test_transpose(self):
    r = np.random.random((9, 1, 23, 4, 10, 16))
    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for dim in [(0, 1), (1, 0), (0, 2), (1, 2), (2, 0), (2, 1), (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)]:
      axis = [0, 1, 2, 3, 4, 5]
      axis[dim[0]], axis[dim[1]] = axis[dim[1]], axis[dim[0]]
      b = a.transpose(axis)
      tb = ta.transpose(*dim)

      self.__tensor_assert(b, None, TransposeBackward, False, True)
      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      c = b.sum()
      tc = tb.sum()

      c.backward()
      tc.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_max(self):
    # TODO
    pass

  def test_exp(self):
    r = np.random.random((10, 4, 1, 9, 4)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    b = a.exp()
    tb = ta.exp()

    self.__tensor_assert(b, None, ExpBackward, False, True)
    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    c = b.sum()
    tc = tb.sum()

    c.backward()
    tc.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_log(self):
    r = np.random.random((10, 4, 1, 9, 4)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    b = a.log()
    tb = ta.log()

    self.__tensor_assert(b, None, LogBackward, False, True)
    self.__assert_numpy_equals(tb.detach().numpy(), b.data)

    c = b.sum()
    tc = tb.sum()

    c.backward()
    tc.backward()

    self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_pow(self):
    r = np.random.random((10, 4, 1, 9, 4)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for exponent in [1, 2, 3, 4, 1/2, -1, -1/2, 1/4, 1/55]:
      b = a ** exponent
      tb = ta ** exponent

      self.__tensor_assert(b, None, PowBackward, False, True)
      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      c = b.sum()
      tc = tb.sum()

      c.backward()
      tc.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_squeeze(self):
    r = np.random.random((1, 4, 1, 9, 4, 1)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for dim in [0, 2, 5]:
      b = a.squeeze(dim)
      tb = ta.squeeze(dim)

      self.__tensor_assert(b, None, SqueezeBackward, False, True)
      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      c = b.sum()
      tc = tb.sum()

      c.backward()
      tc.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_unsqueeze(self):
    r = np.random.random((4, 9, 4)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for dim in [0, 2, 3]:
      b = a.unsqueeze(dim)
      tb = ta.unsqueeze(dim)

      self.__tensor_assert(b, None, UnsqueezeBackward, False, True)
      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      c = b.sum()
      tc = tb.sum()

      c.backward()
      tc.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_reshape(self):
    r = np.random.random((4, 9, 4)).astype(np.float32)

    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for shape in [(4, 9, 4), (36, 4), (4, 36), (9, 16), (1, 36, 4)]:
      b = a.reshape(*shape)
      tb = ta.reshape(*shape)

      self.__tensor_assert(b, None, ReshapeBackward, False, True)
      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      c = b.sum()
      tc = tb.sum()

      c.backward()
      tc.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  def test_as_strided(self):
    r = np.random.random((4, 9, 4)).astype(np.float32)
    
    a = Tensor(r, requires_grad=True)
    ta = torch.tensor(r, requires_grad=True)

    for shape, strides in [
      ((4, 4), (4, 1)),
      ((4, 9), (36, 4)),
      ((2, 2, 2), (16, 8, 4)),
      ((3, 2, 1, 3), (6, 3, 3, 1))
                           ]:
      b = a.as_strided(shape, strides)
      tb = ta.as_strided(shape, strides)

      self.__tensor_assert(b, None, AsStridedBackward, False, True)
      self.__assert_numpy_equals(tb.detach().numpy(), b.data)

      c = b.sum()
      tc = tb.sum()

      c.backward()
      tc.backward()

      self.__assert_numpy_equals(ta.grad.numpy(), a.grad)

  # Test example from README

  def test_example(self):
    a = Tensor(2.0, requires_grad=False)
    b = Tensor(2.0, requires_grad=False)
    c = a * b
    c.requires_grad = True
    d = Tensor(2.0, requires_grad=False)
    e = c * d
    f = Tensor(2.0, requires_grad=False)
    g = e * f
    h = Tensor(2.0, requires_grad=True)
    i = g / h
    j = i.log()
    k = j + h
    m = k.exp()
    m.backward()
    
    print(m)      # 59.1124
    print(c.grad) # 14.7781
    print(h.grad) # 29.5562
    self.assertAlmostEqual(m.data, 59.1124, places=4)
    self.assertAlmostEqual(c.grad, 14.7781, places=4)
    self.assertAlmostEqual(h.grad, 29.5562, places=4)

  def test_simple_autograd(self):
    a = Tensor(2.0, requires_grad=False)
    self.__tensor_assert(a, None, NoneFn, True, False)
    b = Tensor(2.0, requires_grad=False)
    self.__tensor_assert(b, None, NoneFn, True, False)
    c = a * b
    self.__tensor_assert(c, None, NoneFn, True, False)
    c.requires_grad = True
    self.__tensor_assert(c, None, Accumulate, True, True)
    d = Tensor(2.0, requires_grad=False)
    self.__tensor_assert(d, None, NoneFn, True, False)
    e = c * d
    self.__tensor_assert(e, None, MulBackward, False, True)
    f = Tensor(2.0, requires_grad=False)
    self.__tensor_assert(f, None, NoneFn, True, False)
    g = e * f
    self.__tensor_assert(g, None, MulBackward, False, True)
    h = Tensor(2.0, requires_grad=True)
    self.__tensor_assert(h, None, Accumulate, True, True)
    i = g * h
    self.__tensor_assert(i, None, MulBackward, False, True)
    j = i + h
    self.__tensor_assert(j, None, AddBackward, False, True)
    k = j * i
    self.__tensor_assert(k, None, MulBackward, False, True)
    k.backward()

    self.assertEqual(c.grad, 528)
    self.assertEqual(h.grad, 1088)

    # for x in [a,b,c,d,e,f,g,h,i,j,k]:
    #   print(x.grad)
  