"""
This file contains helpful utility functions
for neural network parameter initialization schemes.
"""
from __future__ import division

import math

import deepscoop.autograd.tensor_library as tl
import deepscoop.autograd.tensor_library.random as tlr

def get_in_out_dims(tensor):
	"""
	Gets the fan-in and fan-out
	dimensions of the provided
	tensor.
	"""
	dims = tensor.dims
	if dims < 2:
		raise ValueError("Tensor must have at least 2 dimensions.")

	if dims == 2:
		nin  = tensor.shape[1]
		nout = tensor.shape[0]
	else:
		num_input_fmaps = tensor.shape[1]
		num_output_fmaps = tensor.shape[0]
		receptive_field_size = 1
		if dims > 2:
			receptive_field_size = tensor[0][0]._data.size
		nin  = num_input_fmaps * receptive_field_size
		nout = num_output_fmaps * receptive_field_size

	return nin, nout

def normalized(nin, nout):
	"""
	See Glorot and Bengio (2010)
	Pg. 253, Eq. (16)
	"""
	# nin, nout = get_in_out_dims(tensor)
	high = tl.sqrt(6 / float(nin + nout))
	low = -high
	return tlr.uniform(low, high, size=(nin, nout))

def glorot_uniform(nin, nout):
	# nin, nout = get_in_out_dims(tensor)
	high = tl.sqrt(3 / nin)
	low = -high
	return tlr.uniform(low, high, size=(nin, nout))

def xavier_normal(nin, nout):
	# nin, nout = get_in_out_dims(tensor)
	std = tl.sqrt(2.0 / float(nin + nout))
	return tlr.normal(scale=std, size=(nin, nout))

def sparse(tensor, sparsity, std=0.01):
	"""
	Initialization method described in
	"Deep Learning via Hessian-Free Optimization"
	by Martens, J. (2010).
	"""
	if tensor.dims != 2:
		raise ValueError("This initialization only works with Tensors having 2 dimensions.")

	rows, cols = tensor.shape
	num_zeros = int(math.ceil(sparsity * rows))

	X = tlr.normal(loc=0, scale=std)
	for col_idx in range(cols):
		row_idxs = tlr.permutation(rows)
		zero_idxs = row_idxs[:num_zeros]
		X[zero_idxs, col_idx] = 0.
	return X
