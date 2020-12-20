from utils import topological_sort
import inspect
import operator
import numpy as np

def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        unary_fun = lambda x: fun(*subval(args, argnum, x), **kwargs)
        vjp, ans = make_grad(unary_fun, args[argnum])
        return vjp(np.ones_like(ans))
    return gradfun

def make_grad(fn, x):
    start_node = GradNode.new_root()
    end_value, end_node =  trace(start_node, fn, x)
    if end_node:
        def grad_fn(g): return backward(g, end_node)
    return grad_fn, end_value

def backward(x, end_node):
    outgrads = {end_node : x}
    for node in topological_sort(end_node):
        g = outgrads.pop(node)
        fun, value, args, kwargs, argnums = node.package
        for argnum, parent in zip(argnums, node.parents):
            grad = node.grad_fns[argnum]
            parent_grad = g * grad(value, *args, **kwargs)
            outgrads[parent] = my_sum(outgrads.get(parent), parent_grad)
    return g 

def my_sum(y, x):
    if y is None:
        return x
    return y + x

def trace(start_node, fn, x):
    start_box = Tensor(x, start_node)
    end_box = fn(start_box)
    return end_box._data, end_box._node
