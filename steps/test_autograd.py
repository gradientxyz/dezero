if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from dezero import Variable
from dezero.core_simple import pow


def numerical_diff(f, x, eps=1e-4, *param):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0, *param)
    y1 = f(x1, *param)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = x ** 2
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = x ** 2
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = x ** 2
        y.backward()
        num_grad = numerical_diff(pow, x, 2)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

unittest.main()