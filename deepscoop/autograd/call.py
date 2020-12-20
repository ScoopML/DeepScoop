import inspect

import flamethrower.autograd.tensor as ten
import flamethrower.autograd.variable as var
import flamethrower.autograd.node as anode
import flamethrower.autograd.grad_defs as gd

from .utils import name

def isvar(x):
	return isinstance(x, var.Variable)

def get_data(args, var_args):
	args = list(args)
	for i, arg in var_args:
		args[i] = arg._data
	return tuple(args)

def get_parents(var_args):
	if hasattr(var_args, '__iter__'):
		return tuple(x[1]._node for x in var_args)
	else:
		return tuple(x[1]._node)

def get_argnums(var_args):
	if hasattr(var_args, '__iter__'):
		return tuple(x[0] for x in var_args)
	else:
		return tuple(x[0])

def find_var_args(args):
	var_args = []
	node_type = None
	for argnum, arg in enumerate(args):
		if isvar(arg):
			var_args.append((argnum, arg))
			node_type = type(arg.node)
	return var_args, node_type

def primitive(fn):
	def f_wrapped(*args, **kwargs):
		var_args, node_type = find_var_args(args)
		if var_args:
			argdata = get_data(args, var_args)
			parents = get_parents(var_args)
			argnums = get_argnums(var_args)
			ans     = fn(*argdata, **kwargs)
			if name(fn) in gd.no_trace_primitives:
				node = anode.NoGradNode(ans, fn, argdata, kwargs, argnums, parents)
				return ten.Tensor(ans, node)
			else:
				node = anode.GradNode(ans, fn, argdata, kwargs, argnums, parents)
				return ten.Tensor(ans, node)
		else:
			try:
				return fn(*args, **kwargs)
			except TypeError:
				return fn(fn, *args, **kwargs)
	f_wrapped.__name__ = fn.__name__
	return f_wrapped

class Primitive2(object):
	def __init__(self, fn):
		self.fn = fn

	def __name__(self):
		try:
			return self.fn.__name__
		except AttributeError:
			return "Flamethrower primitive with undefined name"

	def __call__(self, *args, **kwargs):
		var_args, node_type = find_var_args(args)
		if var_args:
			argdata = get_data(args, var_args)
			parents = get_parents(var_args)
			argnums = get_argnums(var_args)
			ans     = self(*argdata, **kwargs)
			if name(self.fn) in gd.no_trace_primitives:
				node = anode.NoGradNode(ans, self, argdata, kwargs, argnums, parents)
				return ten.Tensor(ans, node)
			else:
				node = anode.GradNode(ans, self, argdata, kwargs, argnums, parents)
				return ten.Tensor(ans, node)
		else:
			try:
				return self.fn(*args, **kwargs)
			except TypeError:
				return self.fn(self.fn, *args, **kwargs)

