import deepscoop.autograd.tensor_library as tl

class _LRScheduler(object):
	"""
	Base class for learning rate schedulers.
	"""
	def __init__(self, initial_lr, scale=1, iters_per_epoch=None):
		self.iter = 0
		self.iters_per_epoch = iters_per_epoch
		self.epoch = 1
		self.initial_lr = initial_lr
		self.lr = initial_lr
		self.scale = scale

	def step(self):
		raise NotImplementedError

	def reset(self):
		self.iter = 1
		self.epoch = 1
		self.lr = self.initial_lr

class LinearScheduler(_LRScheduler):
	"""
	Implements a linear decay learning rate schedule.
	In other words, alpha_n = alpha_0 / (cn) where n is the current
	iteration.
	"""
	def step(self):
		self.iter += 1
		self.lr = self.initial_lr / (self.scale * self.iter)
		if self.batch_size and (self.iter % self.iters_per_epoch == 0):
			self.epoch += 1
		return self.lr

class ExponentialScheduler(_LRScheduler):
	"""
	Implements an exponential decay learning rate
	schedule. In other words, alpha_n = alpha_0 * (base)^{-cn}
	"""
	def __init__(self, initial_lr, base=tl.exp(1), scale=1, iters_per_epoch=None):
		super(ExponentialScheduler, self).__init__(initial_lr, scale, iters_per_epoch)
		self.base = base

	def step(self):
		self.iter += 1
		self.lr = self.initial_lr * (self.base ** (-self.scale * self.iter))
		return self.lr

class FactorialScheduler(_LRScheduler):
	"""
	Implements a learning rate schedule with factorial decay.
	In other words, alpha_n = alpha_0 / (n!)
	"""
	def step(self):
		self.iter += 1
		self.lr /= self.iter
		return self.lr

class TrigonometricScheduler(_LRScheduler):
	"""
	Implements a learning rate schedule with trigonometric
	oscillation. In other words, alpha_n = alpha_0 * trig_fn(c * n)
	"""
	def __init__(self, initial_lr, trig_fn=tl.cos, scale=1, iters_per_epoch=None):
		super(TrigonometricScheduler, self).__init__(initial_lr, scale, iters_per_epoch)
		self.trig_fn = trig_fn

	def step(self):
		self.iter += 1
		self.lr = self.initial_lr * self.trig_fn(self.scale * self.iter)
		return self.lr

