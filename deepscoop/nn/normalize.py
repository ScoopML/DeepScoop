import flamethrower.autograd.tensor_library as tl
from flamethrower.autograd import Tensor
from .module import Module

class BatchNorm(Module):
	def __init__(self, eps=1e-5, gamma=1, beta=0):
		super(BatchNorm, self).__init__()
		self.eps = eps
		self.gamma = Tensor(gamma)
		self.new_param("gamma", self.gamma)
		self.beta = Tensor(beta)
		self.new_param("beta", self.beta)

	def forward(self, X):
		n = X.shape[0]
		mu = tl.sum(X, axis=0) / n
		std = tl.sum((X - mu) ** 2, axis=0) / n
		x_hat = (X - mu) / tl.sqrt(std + self.eps)
		y = self.gamma * x_hat + self.beta
		return y
