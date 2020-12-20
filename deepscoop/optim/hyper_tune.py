import numpy as np
import flamethrower.autograd.tensor_library as tl

def random_search(rng, shape, discrete=False):
	if not discrete:
		generated = tl.random.uniform(*rng, size=shape)
	else:
		generated = tl.random.choice(tl.array(rng), size=shape)
	return generated

