import flamethrower.autograd.tensor_library as tl
from .module import Module
from .utils import get_logger

import logging

logger = get_logger()

class Dropout(Module):
	def __init__(self, p=0.5, on=True):
		super(Dropout, self).__init__()
		self.p = p
		self.on = on

	def test_mode(self):
		self.on = False

	def train_mode(self):
		self.on = True

	def forward(self, X):
		logger.info(f"Using dropout on data: {X} with probability: {self.p}")
		if not self.on:
			return X
		mask = tl.random.uniform(0, 1, size=X.shape)
		mask = mask < self.p
		return X * mask

class L2Regularizer(Module):
	def __init__(self, weights, scale=1):
		super(L2Regularizer, self).__init__()
		try:
			iter(weights)
		except TypeError:
			weights = [weights]
		self.weights = weights
		self.scale = scale

	def forward(self):
		logger.info(f"Using L2 Regularization")
		term = 0
		for w in self.weights:
			term += tl.sum(tl.square(w))
		return self.scale * term

class L1Regularizer(Module):
	def __init__(self, weights, scale=1):
		super(L1Regularizer, self).__init__()
		try:
			iter(weights)
		except TypeError:
			weights = [weights]
		self.weights = weights
		self.scale = scale

	def forward(self):
		logger.info(f"Using L1 Regularization")
		term = 0
		for w in self.weights:
			term += tl.sum(tl.abs(w))
		return self.scale * term

class ElasticNetRegularizer(Module):
	def __init__(self, weights, lambda1=0.5, lambda2=0.5):
		super(ElasticNetRegularizer, self).__init__()
		try:
			iter(weights)
		except TypeError:
			weights = [weights]
		self.weights = weights
		self.lambda1 = lambda1
		self.lambda2 = lambda2

	def forward(self):
		logger.info("Using elastic net regularization.")
		term = 0
		for w in self.weights:
			term += self.lambda1 * tl.sum(tl.abs(w)) \
			      + self.lambda2 * tl.sum(tl.square(w))
		return term

def label_smoother(labels, eps=0.05):
	K = len(labels)
	labels[labels == 0] = eps / (K - 1)
	labels[labels == 1] = 1 - eps
	return labels


		
