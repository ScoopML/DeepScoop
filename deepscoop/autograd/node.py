import deepscoop.autograd.grad_defs as gd
import deepscoop.autograd.utils as utils

class Node(object):
	"""
	Base class for a generic
	node type.
	"""
	__slots__ = []
	def __init__(self, val, fn, args, kwargs, argnums, parents):
		raise NotImplementedError

	def init_root(self, *args, **kwargs):
		raise NotImplementedError

	@property
	def is_root(self):
		return self._is_root

	@classmethod
	def new_root(cls, *args, **kwargs):
		root = cls.__new__(cls)
		root.init_root(*args, **kwargs)
		root._is_root = True
		return root

class GradNode(Node):
	"""
	A node that attaches to a `Variable`
	for which we want to take a gradient.
	"""
	__slots__ = ['name', 'parents', '_grad', 'grad_fns', 'package', '_is_root']
	def __init__(self, val, fn, args, kwargs, argnums, parents, node_name=None):
		self._is_root = False
		self.parents = parents
		self.package = fn, val, args, kwargs, argnums
		self.name = node_name
		try:
			self.grad_fns = gd.grad_definitions[utils.name(fn)]
		except KeyError:
			fn_name = utils.name(fn) 
			raise NotImplementedError("Grad of {} wrt argnums {} not defined".format(fn_name, argnums))

	@property
	def grad(self):
		return self._grad
	
	def __str__(self):
		if self.name:
			return self.name
		else:
			return "Unnamed GradNode Object"

	def init_root(self):
		self.parents = []
		self.package = (lambda x: x, None, (), {}, [])
		self.grad_fns = {0: lambda g, ans, x, y: (), 1: lambda g, ans, x, y: ()}

class NoGradNode(Node):
	"""
	A node that attaches to a `Variable`
	for which we DO NOT want to take a gradient.
	"""
	__slots__ = ['name', 'parents', 'package', '_is_root']
	def __init__(self, val, fn, args, kwargs, argnums, parents, node_name=None):
		self._is_root = False
		self.parents = parents
		self.package = fn, val, args, kwargs, argnums
		self.name = node_name

	def __str__(self):
		if self.name:
			return self.name
		else:
			return "Unnamed GradNode Object"

	def init_root(self):
		self.parents = []
		self.package = (lambda x: x, None, (), {}, [])
