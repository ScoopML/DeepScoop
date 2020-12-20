from __future__ import absolute_import

import flamethrower.autograd.variable as var
import flamethrower.autograd.grad_defs as gd
import flamethrower.autograd.tensor_library as tl 

import numpy as np

class Tensor(var.Variable):
	__array_priority__ = 100.0

	__getitem__ = gd.container_take

	def __len__(self): return len(self._data)

	def __neg__(self): return tl.negative(self)
	def __add__(self, other): return tl.add(self, other)
	def __radd__(self, other): return tl.add(other, self)
	def __sub__(self, other): return tl.subtract(self, other)
	def __rsub__(self, other): return tl.subtract(other, self)
	def __mul__(self, other): return tl.multiply(self, other)
	def __rmul__(self, other): return tl.multiply(other, self)
	def __div__(self, other): return tl.divide(self, other)
	def __rdiv__(self, other): return tl.divide(other, self)
	def __pow__(self, other): return tl.power(self, other)
	def __rpow__(self, other): return tl.power(other, self)
	def __mod__(self, other): return tl.mod(self, other)
	def __rmod__(self, other): return tl.mod(other, self)
	def __truediv__(self, other): return tl.true_divide(self, other)
	def __rtruediv__(self, other): return tl.true_divide(other, self)
	def __matmul__(self, other): return tl.matmul(self, other)
	def __rmatmul__(self, other): return tl.matmul(other, self)
	def __eq__(self, other): return tl.equal(self, other)
	def __ne__(self, other): return tl.not_equal(self, other)
	def __gt__(self, other): return tl.greater(self, other)
	def __ge__(self, other): return tl.greater_equal(self, other)
	def __lt__(self, other): return tl.less(self, other)
	def __le__(self, other): return tl.less_equal(self, other)
	def __abs__(self): return tl.abs(self)
	def __hash__(self): return id(self)


tensor_types = [float, np.float16, np.float32, np.float64,
				complex, np.complex64, np.complex128, np.ndarray]
for _type in tensor_types:
	Tensor.register(_type)
