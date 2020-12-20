import flamethrower.autograd as ag
import numpy as np
import logging

class Optimizer(object):
	"""
	Abstract base class for neural network optimizers.
	"""
	def __init__(self, params, defaults, name=None):
		self.params = params
		self.defaults = defaults
		self.name = name
		if not isinstance(params, ag.Tensor):
			logging.error("Trying to initialize params with non-tensor type.")
			raise TypeError("Params should be an iterable of Tensors.")

		self.state = {}
		self.param_groups = []


		param_groups = list(params)
		if len(param_groups) == 0:
			raise ValueError("Empty parameter list.")
		if not isinstance(param_groups[0], dict):
			param_groups = [{'params': param_groups}]

		for pg in param_groups:
			self.add_param_group(pg)

	def add_param_group(self, param_group):
		r"""Add a param group to the :class:`Optimizer` s `param_groups`.

		This can be useful when fine tuning a pre-trained network as frozen layers can be made
		trainable and added to the :class:`Optimizer` as training progresses.

		Arguments:
			param_group (dict): Specifies what Tensors should be optimized along with group
			specific optimization options.
		"""
		self.param_groups.append(param_group)

	def __getstate__(self):
		return {
			'defaults': self.defaults,
			'state': self.state,
			'params': self.param_groups
		}

	def __setstate__(self, state):
		logging.info(f"Setting state: {state}")
		self.__dict__.update(state)

	def step(self, closure=None):
		raise NotImplementedError
