import unittest
from gradflow.autograd import unbroadcast
import numpy as np

class TestBroadcast(unittest.TestCase):
  def test_unbroadcast2D(self):
    a = np.random.randn(3, 4)
    b = np.random.randn(3, 1)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(1, 4)

    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(1, 1)

    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(4)

    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)
  
  def test_unbroadcast3D(self):
    a = np.random.randn(3, 4, 5)

    b = np.random.randn(3, 1, 5)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(1, 4, 5)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(1, 1, 5)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(1, 1, 1)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(1, 4, 1)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(4, 5)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(4, 1)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

    b = np.random.randn(3, 4, 1)
    c = a + b

    self.assertTrue(unbroadcast(c, a.shape).shape == a.shape)

  def test_unbroadcastND(self):
    # TODO
    pass



# a = TestBroadcast()
# a.test_unbroadcast2D()