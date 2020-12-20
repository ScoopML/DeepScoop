from .module import Module
from deepscoop.autograd import Tensor

import deepscoop.autograd.tensor_library as tl
import deepscoop.autograd.tensor_library.random as tlr
import deepscoop.nn.initialize as init

import logging

logger = logging.getLogger(__name__)

class Linear(Module):
	def __init__(self, in_size, out_size, use_bias=True, init_fn=None):
		logger.info(f"Creating a Linear layer with dimensions ({in_size}, {out_size})")
		logger.info(f"Using bias: {use_bias}")
		super(Linear, self).__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.use_bias = use_bias
		self._init_params(init_fn=init_fn)

	def _init_params(self, init_fn=None):
		if not init_fn:
			init_fn = init.glorot_uniform
		self.W = Tensor(init_fn(self.out_size, self.in_size))
		self.new_param('W', self.W)
		if self.use_bias:
			self.b = Tensor(init_fn(1, self.W.shape[0]))
			self.new_param('b', self.b)

	def forward(self, X):
		logger.info(f"Running forward pass on data: {X.data}")
		out = X @ tl.transpose(self.W)
		if self.use_bias:
			out += self.b
		return out
