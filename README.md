

<p align="center">
    <img alt="GitHub Profile Readme Generator" src="./img/ScoopDeep.png" width="200" height="200"/>
</p>

<h1 align="center">
</h1>

<p align="center">
<a href="https://github.com/ScoopML/DeepScoop/blob/main/LICENSE" target="blank">
<img src="https://img.shields.io/github/license/ScoopML/DeepScoop?style=flat-square" alt="DeepScoop licence" />
</a>
<a href="https://github.com/ScoopML/DeepScoop/fork" target="blank">
<img src="https://img.shields.io/github/forks/ScoopML/DeepScoop?style=flat-square" alt="DeepScoop forks"/>
</a>
<a href="https://github.com/ScoopML/DeepScoop/stargazers" target="blank">
<img src="https://img.shields.io/github/stars/ScoopML/DeepScoop?style=flat-square" alt="DeepScoop stars"/>
</a>
<a href="https://github.com/ScoopML/DeepScoop/issues" target="blank">
<img src="https://img.shields.io/github/issues/ScoopML/DeepScoop?style=flat-square" alt="DeepScoop issues"/>
</a>
<a href="https://github.com/ScoopML/DeepScoop/pulls" target="blank">
<img src="https://img.shields.io/github/issues-pr/ScoopML/DeepScoop?style=flat-square" alt="DeepScoop pull-requests"/>
</a>
</p>

## Quick Start

Install it:

```bash
pip install oreoweb
```

Basic Usage:

```python
from __future__ import division

import numpy as np
import sys

import deepscoop.nn as nn
import deepscoop.nn.loss as loss
import deepscoop.optim as optim
import deepscoop.autograd.tensor_library as tl

from deepscoop.autograd import Tensor
import matplotlib.pyplot as plt

from download_mnist import load_mnist
from feedforward import FeedforwardNetwork
import deepscoop.nn.utils as utils

utils.configure_logging()

np.random.seed(0)

X, y, X_test, y_test = load_mnist()

NUM_EXAMPLES = len(y)
DATA_SIZE = 784
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10
NUM_ITERS = NUM_EXAMPLES // BATCH_SIZE
eps = 1e-8

X = X.reshape(len(X), DATA_SIZE)
X = np.divide(X, 255)
X = X - np.mean(X, axis=0, keepdims=True)
z = np.std(X, axis=1, keepdims=True)
z[z == 0] = eps
X /= z

X = Tensor(X)
model = FeedforwardNetwork(DATA_SIZE, NUM_CLASSES)
optimizer = optim.SGD(model.params(), lr=1e-3)
loss_fn = loss.cross_entropy

for epoch in range(NUM_EPOCHS):
	print(f"### Current Epoch: {epoch} ###")
	for i in range(NUM_ITERS):
		if i % 50 == 0 and i != 0:
			print(f"iter: {i}, loss: {loss.data}")
		sidx = i * BATCH_SIZE
		eidx = (i + 1) * BATCH_SIZE
		batch = X[sidx:eidx]
		preds = model(batch)
		y_hat = y[sidx:eidx]
		loss = loss_fn(y_hat, preds)
		loss.backward()
		optimizer.step()

```



