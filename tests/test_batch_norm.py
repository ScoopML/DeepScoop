import deepscoop.nn.normalize as norm
import deepscoop.nn.activations as act
import deepscoop.nn as nn

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.lin1 = nn.Linear(100, 200)
		self.lin2 = nn.Linear(200, 4)
		self.bn = norm.BatchNorm()

	def forward(self, X):
		y = self.bn(self.lin1(X))
		z = act.softmax(self.lin2(y))
		return z

if __name__ == '__main__':
	import numpy as np
	X = np.random.randn(400, 100)
	m = Model()
	z = m(X)
