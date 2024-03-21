import unittest
from gradflow import *
import gradflow.functions as F
import numpy as np
import torch
import torch.nn as nn

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
    # TODO: test more dimensions
    r1 = np.random.random((4, 9)).astype(np.float32)
    r2 = np.random.random((9, 4)).astype(np.float32)

    a = Tensor(r1, requires_grad=True)
    b = Tensor(r2, requires_grad=True)

    ta = torch.tensor(r1, requires_grad=True)
    tb = torch.tensor(r2, requires_grad=True)

    c = a @ b
    tc = ta @ tb

    self.__tensor_assert(c, None, MatmulBackward, False, True)
    self.__assert_numpy_equals(tc.detach().numpy(), c.data)

    d = c.mean()
    td = tc.mean()

    d.backward()
    td.backward()

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
      #...
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
  
#   def test_relu(self):
#     relu = nn.ReLU()
#     arelu = F.ReLU()

#     data = np.arange(9).reshape(3, 3).astype(np.float32)
#     a = torch.tensor(data, requires_grad=True)
#     b = a * -2
#     c = b.T
#     d = relu(a + c)
#     e = d.mean()
#     e.backward()

#     aa = Tensor(data, requires_grad=True)
#     ab = aa * -2
#     ac = ab.transpose()
#     ad = arelu(aa + ac)
#     ae = ad.mean()
#     ae.backward()

#     self.__assert_numpy_equals(a.grad.numpy(), aa.grad)
  
#   # def test_reductions(self):
#   #   rnd0 = np.random.random((3, 3))
#   #   rnd1 = np.random.random((3, 3))

#   #   a = torch.tensor(rnd0, requires_grad=True)
#   #   b = torch.tensor(rnd1, requires_grad=True)

#   #   c = a / b
#   #   c.sum().backward()

#   #   aa = Tensor(rnd0, requires_grad=True)
#   #   ab = Tensor(rnd1, requires_grad=True)

#   #   ac = aa / ab
#   #   ac.sum().backward()

#   #   self.__assert_numpy_equals(a.grad.numpy(), aa.grad)
#   #   self.__assert_numpy_equals(b.grad.numpy(), ab.grad)

#   #   a = torch.tensor(rnd0, requires_grad=True)
#   #   b = torch.tensor(rnd1, requires_grad=True)

#   #   c = a / b
#   #   c.mean().backward()

#   #   aa = Tensor(rnd0, requires_grad=True)
#   #   ab = Tensor(rnd1, requires_grad=True)

#   #   ac = aa / ab
#   #   ac.mean().backward()

#   #   print(a.mean(), aa.mean())
#   #   self.__assert_numpy_equals(a.grad.numpy(), aa.grad)
#   #   self.__assert_numpy_equals(b.grad.numpy(), ab.grad)
  
#   def test_loss(self):
#     mse = nn.MSELoss()

#     a = torch.tensor([1, 0.4, -12, -0.314], requires_grad=True)
#     b = torch.tensor([0.001, 2, 10, 3.14], requires_grad=True)

#     z = mse(a, b)
#     z.backward()
  
#     mmse = F.MSELoss()

#     ma = Tensor([1, 0.4, -12, -0.314], requires_grad=True)
#     mb = Tensor([0.001, 2, 10, 3.14], requires_grad=True)

#     mz = mmse(ma, mb)
#     mz.backward()

#     self.__assert_numpy_equals(a.grad.numpy(), ma.grad)
#     self.__assert_numpy_equals(b.grad.numpy(), mb.grad)
  
#   # def test_classification_loss(self):
#   #   preds = np.random.randn(3, 5)
#   #   target = np.array([0, 1, 5]) # 0 <= x < C = 5

#   #   print(f"{preds=}")
#   #   print(f"{target=}")

#   #   tpreds = torch.tensor(preds, requires_grad=True)
#   #   ttarget = torch.tensor(target)
#   #   gpreds = Tensor(preds, requires_grad=True)
#   #   gtarget = Tensor(target)

#   #   tlogsoft = nn.LogSoftmax(dim=-1)
#   #   glogsoft = F.LogSoftmax(dim=-1)

#   #   tlogpreds = tlogsoft(tpreds)
#   #   glogpreds = glogsoft(gpreds)

#   #   print(f"{tlogpreds=}")
#   #   print(f"{glogpreds=}")

#   #   tcriterion = nn.NLLLoss()
#   #   gcriterion = F.NLLLoss()
#   #   # TODO: crossentropyloss

#   #   tloss = tcriterion(tlogpreds, ttarget)
#   #   gloss = gcriterion(glogpreds, gtarget)

#   #   print(f"{tloss=}")
#   #   print(f"{gloss=}")

# if __name__ == "__main__":
#   unittest.main()
