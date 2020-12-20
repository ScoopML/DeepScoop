import deepscoop.nn as nn
import deepscoop.autograd as ag
import deepscoop.autograd.utils as utils

from deepscoop.autograd import Tensor

def f(x):
	return x ** 2

x = Tensor([1, 2, 3])
y = f(x)

accurate = utils.grad_check(f, x)
