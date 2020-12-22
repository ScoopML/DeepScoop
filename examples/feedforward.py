import numpy as np

import deepscoop.nn as nn
import deepscoop.optim as optim
import deepscoop.autograd.tensor_library as tl

from deepscoop.autograd import Tensor
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.linspace(-10, 10, num=100)
y_hat = np.sin(X)

X_train = Tensor(X)

class Network(nn.Module):
	def __init__(self, num_in, num_out):
		super(Network, self).__init__()
		self.num_in = num_in
		self.num_out = num_out
		self.lin1 = nn.Linear(num_in, 5)
		self.lin2 = nn.Linear(5, num_out)
		self.add_module('linear1', self.lin1)
		self.add_module('linear2', self.lin2)

	def _forward(self, X):
		out = self.lin1(X)
		out.node().name = 'W_1 @ X'
		out = self.lin2(out)
		out.node().name = 'W_2 @ X'
		return out

model = Network(1, 1)
optimizer = optim.SGD(model.params(), lr=1e-4)
num_iters = 1000
loss_fn = lambda y, y_hat: (y_hat - y) ** 2

for i in range(num_iters):
	print(i)
	break
	batch = Tensor([X[i % len(X)]])
	y = model(batch)
	loss = loss_fn(y, [y_hat[i % len(y_hat)]])
	loss.node().name = '(y_hat - y)^2'
	# loss.data().resize(100, 1)
	loss.backward()
	optimizer.step()
	# optimizer.zero_grad()
	# print("W1 grad is: {} ".format(model.lin1.W.node().grad))

out = []
for j in range(len(y_hat)):
	z = model([X[j]]).data()[0]
	out.append(z)
out = np.array(out)
plt.plot(X, out)
