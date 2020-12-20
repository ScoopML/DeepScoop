"""
A collection of loss
functions used in training
deep neural networks.
"""
from __future__ import division
import flamethrower.autograd.tensor_library as tl

def cross_entropy(y, y_hat, regularizer=None):
	"""
	Cross-entropy loss, used for
	classification with n class.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	total = 0
	z = log_softmax(y_hat, axis=1)
	for i, label in enumerate(y):
		total += -z[i][label]
	return total / len(y) + regularizer()

def log_softmax(x, axis=None):
	b = tl.max(x, axis=axis, keepdims=True)
	return (x - b) - tl.log(tl.sum(tl.exp(x - b), axis=axis, keepdims=True))
	
def binary_cross_entropy(y, y_hat, eps=1e-5, regularizer=None):
	"""
	Binary cross-entropy loss, used
	for classification with 2 classes.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return tl.mean(-y * tl.log(y_hat + eps) - (1 - y) * tl.log(1 - y_hat + eps)) + regularizer()

def mean_squared_error(y, y_hat, regularizer=None):
	"""
	MSE loss, often used for
	regression tasks.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	n = len(y)
	l = (1 / n) * tl.sum((y_hat - y) ** 2)
	return l + regularizer()

def l2(y, y_hat, regularizer=None):
	"""
	Loss function using the L2 norm.
	Equivalent to minimization with
	MSE loss.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return tl.sum(tl.sqrt((y_hat - y) ** 2)) + regularizer()

def l1(y, y_hat, regularizer=None):
	"""
	Loss function using the L1 norm.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return tl.abs(y_hat - y) + regularizer()

def kl_divergence(p, q, regularizer=None):
	"""
	Returns a notion of "distance"
	between two probability distributions
	p and q.
	"""
	if regularizer is None:
		regularizer = lambda: 0
	return -tl.sum(p * tl.log(q / p)) + regularizer()

def huber(y, y_hat, delta=1, regularizer=None):
	if regularizer is None:
		regularizer = lambda: 0
	if tl.abs(y_hat - y) < delta:
		return .5 * (y_hat - y) ** 2 + regularizer()
	else:
		return delta * tl.abs(y_hat - y) - (delta / 2) + regularizer()

def huber_binary_loss(y, y_hat, delta=1, regularizer=None):
	if y_hat == 0:
		y_hat = -1
	if y == 0:
		y = -1
	if regularizer is None:
		regularizer = lambda: 0
	if y_hat * y >= -1:
		return tl.max(0, 1 - y_hat * y) ** 2
	else:
		return -4 * y_hat * y

