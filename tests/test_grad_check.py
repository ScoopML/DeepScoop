import flamethrower.nn as nn
import flamethrower.autograd as ag
import flamethrower.autograd.utils as utils

from flamethrower.autograd import Tensor

def f(x):
	return x ** 2

x = Tensor([1, 2, 3])
y = f(x)

accurate = utils.grad_check(f, x)
