from __future__ import division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
import matplotlib.pyplot as plt

from download_mnist import load_mnist
from feedforward_pt import FeedforwardNetwork

np.random.seed(0)

X, y, X_test, y_test = load_mnist()

NUM_EXAMPLES = len(y)
DATA_SIZE = 784
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10
NUM_ITERS = NUM_EXAMPLES // BATCH_SIZE

X = X.reshape(len(X), DATA_SIZE)
X = np.divide(X, 255)
X = X - np.mean(X, axis=0, keepdims=True)
z = np.std(X, axis=0, keepdims=True)
z[z == 0] = 1
X = X / z
model = FeedforwardNetwork(DATA_SIZE, NUM_CLASSES)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X1 = Tensor(X)
X2 = ag.Tensor(X)
for epoch in range(NUM_EPOCHS):
	print(f"Current Epoch: {epoch}")
	for i in range(NUM_ITERS):
		if i % 50 == 0 and i != 0:
			print(f"Current iteration: {i}")
			print(f"Current loss: {loss.data}")
		sidx = i * BATCH_SIZE
		eidx = (i + 1) * BATCH_SIZE
		batch = X1[sidx:eidx]
		preds = model(batch)
		y_hat = torch.LongTensor(y[sidx:eidx])
		lossval = loss_fn(preds, y_hat)
		lossval.backward()
		optimizer.step()
		break
	break
