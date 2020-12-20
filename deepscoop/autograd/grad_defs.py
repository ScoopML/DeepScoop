from itertools import count

import flamethrower.autograd.utils as utils
import flamethrower.autograd.call as call

import numpy as np

grad_definitions = {}
no_trace_primitives = {}
grad_defs_by_parity = {}

nograd_functions = [
	np.floor, np.ceil, np.round, np.rint, np.around, np.fix, np.trunc, np.all,
	np.any, np.argmax, np.argmin, np.argpartition, np.argsort, np.argwhere, np.nonzero,
	np.flatnonzero, np.count_nonzero, np.searchsorted, np.sign, np.ndim, np.shape,
	np.floor_divide, np.logical_and, np.logical_or, np.logical_not, np.logical_xor,
	np.isfinite, np.isinf, np.isnan, np.isneginf, np.isposinf, np.allclose, np.isclose,
	np.array_equal, np.array_equiv, np.greater, np.greater_equal, np.less, np.less_equal,
	np.equal, np.not_equal, np.iscomplexobj, np.iscomplex, np.size, np.isscalar,
	np.isreal, np.zeros_like, np.ones_like, np.result_type]

for fn in nograd_functions:
	no_trace_primitives[utils.name(fn)] = True

def define_grad(fn, *grads, **kwargs):
	for argnum, partial in enumerate(grads):
		if utils.name(fn) not in grad_definitions:
			grad_definitions[utils.name(fn)] = {}
		grad_definitions[utils.name(fn)][argnum] = partial

def parity(fn):
	return fn.__code__.co_argcount

def record_grad_defs_by_parity(grad_definitions):
	for prim, grad_fn in grad_definitions:
		p = parity(grad_fn)
		if p in grad_defs_by_parity:
			grad_defs_by_parity[p].append(grad_fn)
		else:
			grad_defs_by_parity[p] = [grad_fn]
	return grad_defs_by_parity

def replace_zero(x, val):
	return np.where(x, x, val)
	
# Multivariate gradients
define_grad(np.add,			lambda ans, g, x, y : unbroadcast_f(x, lambda g: g)(g),
							lambda ans, g, x, y : unbroadcast_f(y, lambda g: g)(g))
define_grad(np.subtract,	lambda ans, g, x, y : unbroadcast_f(x, lambda g: g)(g), 
							lambda ans, g, x, y : unbroadcast_f(y, lambda g: -g)(g))
define_grad(np.power,		lambda ans, g, x, y : unbroadcast_f(x, lambda g:   g * y * x ** np.where(y, y - 1, 1.))(g),
							lambda ans, g, x, y : unbroadcast_f(y, lambda g:   g * np.log(replace_zero(x, 1.)) * ans)(g))
define_grad(np.multiply,	lambda ans, g, x, y : unbroadcast_f(x, lambda g:   g * y)(g), 
							lambda ans, g, x, y : unbroadcast_f(y, lambda g:   g * x)(g))
define_grad(np.divide,		lambda ans, g, x, y : unbroadcast_f(x, lambda g:   g / y)(g),
							lambda ans, g, x, y : unbroadcast_f(y, lambda g: - g * x / y**2)(g))
define_grad(np.true_divide,	lambda ans, g, x, y : unbroadcast_f(x, lambda g:   g / y)(g),
							lambda ans, g, x, y : unbroadcast_f(y, lambda g: - g * x / y**2)(g))
define_grad(np.maximum,		lambda ans, g, x, y : unbroadcast_f(x, lambda g:   g * max_min_grad(x, ans, y))(g),
							lambda ans, g, x, y : unbroadcast_f(y, lambda g:   g * max_min_grad(y, ans, x))(g))
define_grad(np.minimum,		lambda ans, g, x, y : unbroadcast_f(x, lambda g:   g * max_min_grad(x, ans, y))(g),
							lambda ans, g, x, y : unbroadcast_f(y, lambda g:   g * max_min_grad(y, ans, x))(g))

# Single variable gradients
define_grad(np.negative,	lambda ans, g, x: -g)
define_grad(np.exp,			lambda ans, g, x: g * ans)
define_grad(np.exp2, 		lambda ans, g, x: g * ans * np.log(2))
define_grad(np.expm1,		lambda ans, g, x: g * (ans + 1))
define_grad(np.log,			lambda ans, g, x: g / x)
define_grad(np.log2,		lambda ans, g, x: g / (x * np.log(2)))
define_grad(np.log10, 		lambda ans, g, x: g / (x * np.log(10)))
define_grad(np.log1p, 		lambda ans, g, x: g / (x + 1))
define_grad(np.sin, 		lambda ans, g, x: g * np.cos(x))
define_grad(np.cos, 		lambda ans, g, x: g * -np.sin(x))
define_grad(np.tan, 		lambda ans, g, x: g / (np.cos(x) ** 2))
define_grad(np.arcsin, 		lambda ans, g, x: g / np.sqrt(1 - x ** 2))
define_grad(np.arccos, 		lambda ans, g, x: -g / np.sqrt(1 - x ** 2))
define_grad(np.arctan, 		lambda ans, g, x: g / np.sqrt(1 + x ** 2))
define_grad(np.sinh, 		lambda ans, g, x: g * np.cosh(x))
define_grad(np.cosh,		lambda ans, g, x: g * np.sinh(x))
define_grad(np.tanh,		lambda ans, g, x: g / (np.cosh(x) ** 2))
define_grad(np.arcsinh, 	lambda ans, g, x: g / np.sqrt(x ** 2 + 1))
define_grad(np.arccosh, 	lambda ans, g, x: g / np.sqrt(x ** 2 - 1))
define_grad(np.arctanh, 	lambda ans, g, x: g / np.sqrt(1 - x ** 2))
define_grad(np.rad2deg, 	lambda ans, g, x: g * 180 / np.pi)
define_grad(np.degrees, 	lambda ans, g, x: g * 180 / np.pi)
define_grad(np.deg2rad, 	lambda ans, g, x: g * np.pi / 180)
define_grad(np.radians, 	lambda ans, g, x: g * np.pi / 180)
define_grad(np.square, 		lambda ans, g, x: g * 2 * x)
define_grad(np.sqrt, 		lambda ans, g, x: g * 0.5 * x ** -0.5)
define_grad(np.sinc, 		lambda ans, g, x: g * ((np.cos(np.pi * x) / x) - (np.sin(np.pi * x))/(np.pi * (x**2))))
define_grad(np.reshape,		lambda ans, g, x, shape, order=None : np.reshape(g, np.shape(x), order=order))
define_grad(np.roll,		lambda ans, g, x, shift, axis=None  : np.roll(g, -shift, axis=axis))

def grad_transpose(ans, g, x, axes=None):
	if axes is not None:
		axes = np.argsort(axes)
	return np.transpose(g, axes)
define_grad(np.transpose, grad_transpose)

def repeat_to_match_shape(g, shape, dtype, axis, keepdims):
	"""Returns the array g repeated along axis to fit vector space vs.
	   Also returns the number of repetitions of the array."""
	if shape == ():
		return g, 1
	axis = list(axis) if isinstance(axis, tuple) else axis
	new_shape = np.array(shape)
	new_shape[axis] = 1
	num_reps = np.prod(np.array(shape)[axis])
	# Can't use broadcast_to because of numpy bug: https://github.com/numpy/numpy/issues/9165
	# return np.broadcast_to(np.reshape(g, new_shape), shape), num_reps
	return np.reshape(g, new_shape) + np.zeros(shape, dtype=dtype), num_reps

def grad_np_sum(ans, g, x, axis=None, keepdims=False, dtype=None):
	shape, dtype = np.shape(x), np.result_type(x)
	return repeat_to_match_shape(g, shape, dtype, axis, keepdims)[0]
define_grad(np.sum, grad_np_sum)

def unbroadcast(x, target_meta, broadcast_idx=0):
	target_shape, target_ndim, target_iscomplex = target_meta
	while np.ndim(x) > target_ndim:
		x = np.sum(x, axis=broadcast_idx)
	for axis, size in enumerate(target_shape):
		if size == 1:
			x = np.sum(x, axis=axis, keepdims=True)
	if np.iscomplexobj(x) and not target_iscomplex:
		x = np.real(x)
	return x

def matmul_adjoint_0(B, G, A_meta, B_ndim):
	if np.ndim(G) == 0:  # A_ndim == B_ndim == 1
		return unbroadcast(G * B, A_meta)
	_, A_ndim, _ = A_meta
	if A_ndim == 1:
		G = np.expand_dims(G, np.ndim(G) - 1)
	if B_ndim == 1:  # The result we need is an outer product
		B = np.expand_dims(B, 0)
		G = np.expand_dims(G, np.ndim(G))
	else:  # We need to swap the last two axes of B
		B = np.swapaxes(B, B_ndim - 2, B_ndim - 1)
	result = np.matmul(G, B)
	return unbroadcast(result, A_meta)

def matmul_adjoint_1(A, G, A_ndim, B_meta):
	if np.ndim(G) == 0:  # A_ndim == B_ndim == 1
		return unbroadcast(G * A, B_meta)
	_, B_ndim, _ = B_meta
	B_is_vec = (B_ndim == 1)
	if B_is_vec:
		G = np.expand_dims(G, np.ndim(G))
	if A_ndim == 1: 
		A = np.expand_dims(A, 1)
		G = np.expand_dims(G, np.ndim(G) - 1)
	else:
		A = np.swapaxes(A, A_ndim - 2, A_ndim - 1)
	result = np.matmul(A, G)
	if B_is_vec:
		result = np.squeeze(result, np.ndim(G) - 1)
	return unbroadcast(result, B_meta)

def matmul_grad_A(ans, g, A, B):
	A_meta = metadata(A)
	B_ndim = np.ndim(B)
	return matmul_adjoint_0(B, g, A_meta, B_ndim)

def matmul_grad_B(ans, g, A, B):
	A_ndim = np.ndim(A)
	B_meta = metadata(B)
	return matmul_adjoint_1(A, g, A_ndim, B_meta)

def metadata(A):
	return np.shape(A), np.ndim(A), np.iscomplexobj(A)

define_grad(np.matmul, matmul_grad_A, matmul_grad_B)

@call.primitive
def container_take(A, idx):
	return A[idx]
def grad_container_take(ans, g, A, idx):
	x = np.zeros_like(A.data)
	np.add.at(x, idx, g)
	return x

define_grad(container_take, grad_container_take)

def unbroadcast(x, target_meta, broadcast_idx=0):
	target_shape, target_ndim, target_iscomplex = target_meta
	while np.ndim(x) > target_ndim:
		x = np.sum(x, axis=broadcast_idx)
	for axis, size in enumerate(target_shape):
		if size == 1:
			x = np.sum(x, axis=axis, keepdims=True)
	if np.iscomplexobj(x) and not target_iscomplex:
		x = np.real(x)
	return x

def unbroadcast_f(target, f):
	target_meta = metadata(target)
	return lambda g: unbroadcast(f(g), target_meta)

def grad_chooser(ans, g, x, axis=None, keepdims=None):
	shape, dtype = np.shape(x), np.result_type(x)
	g_repeated, _ = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
	argmax_locations = x == repeat_to_match_shape(ans, shape, dtype, axis, keepdims)[0]
	return g_repeated * argmax_locations \
		/ np.sum(argmax_locations, axis=axis, keepdims=True)
define_grad(np.max, grad_chooser)

def max_min_grad(x, ans, y):
	if x == ans:
		return 1
	return 0
