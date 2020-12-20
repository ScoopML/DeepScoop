import numpy as anp

def metadata(A):
    return anp.shape(A), anp.ndim(A), anp.result_type(A), anp.iscomplexobj(A)

def unbroadcast(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    while anp.ndim(x) > target_ndim:
        x = anp.sum(x, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = anp.sum(x, axis=axis, keepdims=True)
    if anp.iscomplexobj(x) and not target_iscomplex:
        x = anp.real(x)
    return x

def unbroadcast_f(target, f):
    target_meta = metadata(target)
    return lambda g: unbroadcast(f(g), target_meta)

def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))

fn = lambda ans, x, y : unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y))
