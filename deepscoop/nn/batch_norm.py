import flamethrower.autograd.tensor_library as tl
import flamethrower.autograd.tensor as ten
from .module import Module

class BatchNorm(Module):
	def __init__(self, eps=1e-6, gamma=1, beta=0):
		super(BatchNorm, self).__init__()
		self.eps = eps
		self.gamma = ten.Tensor([gamma])
		self.beta = ten.Tensor([beta])

	def forward(self, X):
		n = X.shape[0]
		mu = tl.sum(X, axis=0) / n
		var = tl.sum((X - mu) ** 2, axis=0) / n
		x_hat = (X - mu) / tl.sqrt(var + self.eps)
		y = self.gamma * x_hat + self.beta
		return y
