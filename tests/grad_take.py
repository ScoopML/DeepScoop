import autograd as ag
import torch
import autograd.numpy as np
import deepscoop.autograd as ag2
import deepscoop.autograd.tensor_library as tl

def with_slice(x):
	y = np.array([1, 2, 3, 2, 1])
	z = x * y
	z = z[1:4]
	w = z + np.array([-1, 2, 1])
	return w

def with_slice2(x):
	y = ag2.Tensor([1, 2, 3, 2, 1])
	z = x * y
	r = z[1:4]
	w = r + ag2.Tensor([2, 1, 4])
	p = tl.sum(w)
	return p, w, r, z, y

def with_slice3(x):
	y = torch.Tensor([1, 2, 3, 2, 1])
	z = x * y
	r = z[1:4]
	w = r + torch.Tensor([2, 1, 4])
	p = torch.sum(w)
	return p, w, r, z, y

x = ag2.Tensor([-1.0, 2.0, -4.0, 7.0, 6.0])
x2 = torch.Tensor(x.data)
x2.requires_grad = True
p2, w2, r2, z2, y2 = with_slice3(x2) 
p, w, r, z, y = with_slice2(x)
p.backward()
p2.backward()
