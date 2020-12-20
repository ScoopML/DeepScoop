import deepscoop.autograd.tensor_library as tl

def sigmoid(x):
	return 1 / (1 + tl.exp(-x))

def softmax(x, axis=1):
	x = x - tl.max(x, axis=axis, keepdims=True)
	y = tl.exp(x)
	return y / tl.sum(y, axis=axis, keepdims=True)

def relu(x):
	return tl.maximum(0, x)

def elu(x, alpha=1.0):
	return relu(x) + tl.mininimum(0, alpha * (tl.exp(x) - 1))

def hardshrink(x, lval=0.5):
	z = tl.copy(x)
	z[(x >= -lval) & (x <= lval)] = 0
	return z

def leaky_relu(x, alpha=0.1):
	return relu(x) + alpha * tl.mininimum(0, x)

def relu6(x):
	return tl.minimum(tl.maximum(0, x), 6)

def rrelu(x, lo=0.125, hi=1/3):
	if x >= 0:
		return x
	return tl.random.uniform(lo, hi) * x

def selu(x, alpha=1.6732632423543772848170429916717):
	return 1.0507009873554804934193349852946 * elu(x, alpha=alpha)

def celu(x, alpha=1.0):
	return relu(x) + tl.minimum(0, alpha * (tl.exp(x / alpha) - 1))

def softplus(x, beta=1, threshold=20):
	if x <= threshold:
		return (1 / beta) * tl.log(1 + tl.exp(beta * x))
	return x

def softsign(x):
	return x / (1 + tl.abs(x))

def tanh(x):
	return tl.tanh(x)

def tanhshrink(x):
	return x - tl.tanh(x)

def threshold(x, val):
	if x > threshold:
		return x
	return val

def softmin(x, axis=1):
	x = -x - tl.max(x, axis=axis, keepdims=True)
	y = tl.exp(x)
	return y / tl.sum(y, axis=axis, keepdims=True)

def log_softmax(x):
	return tl.log(softmax(x))

def hardtanh(x, minx=-1, maxx=1, minv=-1, maxv=1):
	z = tl.copy(x)
	z[z > maxx] = maxv
	z[z < minx] = minv
	return z

def softshrink(x, lval=0.5):
	z = tl.zeros(x.shape)
	z[x > lval] = (x - lval)[x > lval]
	z[x < -lval] = (x + lval)[x < -lval]
	return z
