import deepscoop.autograd.tensor_library as tl

def calc_mi(X, Y, bins):
	counts_XY = tl.histogram2d(X, Y, bins)[0]
	counts_X  = tl.histogram(X, bins)[0]
	counts_Y  = tl.histogram(Y, bins)[0]

	h_X  = entropy(counts_X)
	h_Y  = entropy(counts_Y)
	h_XY = entropy(counts_XY)
	return h_X + h_Y - h_XY

def entropy(counts):
	total = np.sum(counts)
	dist  = counts / total
	dist[dist == 0] = 1
	h = -tl.sum(dist * tl.log(dist))
	return h
