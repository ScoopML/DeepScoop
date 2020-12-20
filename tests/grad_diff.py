import numpy as anp

def grad_diff(ans, a, n=1, axis=-1):
    nd = anp.ndim(a)
    ans_shape = anp.shape(ans)
    sl1 = [slice(None)]*nd
    sl1[axis] = slice(None, 1)

    sl2 = [slice(None)]*nd
    sl2[axis] = slice(-1, None)

    def undiff(g):
        if g.shape[axis] > 0:
            return anp.concatenate((-g[tuple(sl1)], -anp.diff(g, axis=axis), g[tuple(sl2)]), axis=axis)
        shape = list(ans_shape)
        shape[axis] = 1
        return anp.zeros(shape)

    def helper(g, n):
        if n == 0:
            return g
        return helper(undiff(g), n-1)
    return lambda g: helper(g, n)

a = anp.array([1, 2, 4, 7, 0])
ans = anp.array([1, 2, 3, -7])
gradfn = grad_diff(ans, a)
