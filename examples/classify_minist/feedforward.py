from __future__ import division

import numpy as np

import flamethrower.nn as nn
import flamethrower.nn.activations as act
import flamethrower.optim as optim
import flamethrower.autograd.tensor_library as tl


class FeedforwardNetwork(nn.Module):
	def __init__(self, num_in, num_out):
		super(FeedforwardNetwork, self).__init__()
		self.num_in = num_in
		self.num_out = num_out
		self.lin1 = nn.Linear(num_in, 200)
		self.lin2 = nn.Linear(200, 60)
		self.lin3 = nn.Linear(60, num_out)
		self.add_module('linear1', self.lin1)
		self.add_module('linear2', self.lin2)
		self.add_module('linear3', self.lin3)

	def forward(self, X):
		out = act.relu(self.lin1(X))
		out = act.relu(self.lin2(out))
		out = self.lin3(out)
		return out
