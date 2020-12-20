from .utils import get_logger

import deepscoop.autograd.tensor as ten

from collections import OrderedDict
import pickle

logger = get_logger()

class Module(object):
	"""
	Abstract base class for all neural network module
	objects.
	"""
	# Using __slots__ makes things faster
	__slots__ = ['logger', 'is_training', '_buffers', '_params' \
				 '_children', '_state', '_modules']
	def __init__(self, name=None):
		self.name = name
		self.is_training = True
		# Buffers are persistent state objects
		# but not module parameters
		self._buffers = OrderedDict()
		self._params = OrderedDict()
		self._children = OrderedDict()
		self._state = OrderedDict()
		self._modules = OrderedDict()

	def __name__(self):
		return self.name

	def new_buffer(self, name, obj):
		if not isinstance(name, str):
			raise TypeError("Buffer name must be a string.")
		if "." in name:
			raise KeyError("Buffer name may not contain \".\"")
		if name == "":
			raise KeyError("Buffer name must be nonempty.")
		if obj is not None and not isinstance(obj, ten.Tensor):
			raise TypeError("Buffer value must be None or a Tensor, got object of type {}".format(type(obj)))
		if name in self._buffers:
			logger.warning(f"Overriding {name} buffer having value {self._buffers[name]} with {obj}")
		logger.info(f"Creating new buffer with name: {name} and value: {obj}")
		self._buffers[name] = obj

	def named_modules(self, memo=None, prefix=''):
		r"""Returns an iterator over all modules in the network, yielding
		both the name of the module as well as the module itself.

		Yields:
			(string, Module): Tuple of name and module

		Note:
			Duplicate modules are returned only once. In the following
			example, ``l`` will be returned only once.

		Example::

			>>> l = nn.Linear(2, 2)
			>>> net = nn.Sequential(l, l)
			>>> for idx, m in enumerate(net.named_modules()):
					print(idx, '->', m)

			0 -> ('', Sequential(
			  (0): Linear(in_features=2, out_features=2, bias=True)
			  (1): Linear(in_features=2, out_features=2, bias=True)
			))
			1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

		"""

		if memo is None:
			memo = set()
		if self not in memo:
			memo.add(self)
			yield prefix, self
			for name, module in self._modules.items():
				if module is None:
					continue
				submodule_prefix = prefix + ('.' if prefix else '') + name
				for m in module.named_modules(memo, submodule_prefix):
					yield m
		
	def _named_members(self, get_members_fn, prefix='', recurse=True):
		r"""Helper method for yielding various names + members of modules."""
		memo = set()
		modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
		for module_prefix, module in modules:
			members = get_members_fn(module)
			for k, v in members:
				if v is None or str(v) in memo:
					continue
				memo.add(str(v))
				name = module_prefix + ('.' if module_prefix else '') + k
				yield name, v


	def named_parameters(self, prefix='', recurse=True):
		gen = self._named_members(
			lambda module: module._params.items(),
			prefix=prefix, recurse=recurse)
		for elem in gen:
			yield elem

	def new_param(self, name, param):
		logger.info(f"Creating new param with name: {name} and value: {param}")
		self._params[name] = param

	def add_module(self, name, module):
		logger.info(f"Adding module with name: {name} and value: {module} to internal modules.")
		self._modules[name] = module

	def forward(self, *args, **kwargs):
		raise NotImplementedError

	def __call__(self, *args, **kwargs):
		"""
		Applies the forward pass to input arguments
		"""
		logger.info(f"Running forward pass on data: {args}")
		return self.forward(*args, **kwargs)

	def params(self):
		"""
		Returns an iterator over module named parameters
		"""
		for name, param in self.named_parameters():
			yield param

	def children(self):
		"""
		Returns an iterator over named child modules
		"""
		for name, child in self._children:
			yield child

	def set_train_mode(self, train_mode=True):
		"""
		Sets global training mode to train_mode=True
		for self and all child modules
		"""
		logger.info(f"Setting training mode to {train_mode}.")
		self.is_training = train_mode 
		for child in self.children():
			child.set_train_mode(train_mode)
		return self

	def set_infer_mode(self):
		"""
		Sets global training mode to inference (train_mode=False)
		for self and all child modules
		"""
		return self.set_train_mode(train_mode=False)
