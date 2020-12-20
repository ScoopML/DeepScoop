import deepscoop.autograd.tensor as ten

def sum_with_none(x, y):
    if x is None:
        return y
    return x + y

def name(primitive):
    """
    Gets the __name__
    of a `Primitive`.
    """
    try:
        return primitive.__name__()
    except TypeError:
        return primitive.__name__

def topological_sort(end_node):
    """
    Performs a topological start
    beginning with end_node and
    working backwards via parents
    until the start of the computation
    is reached.
    """
    stack = [end_node]
    visited = {}
    topo_sorted = []
    while stack:
        node = stack.pop()
        if node in visited:
            del topo_sorted[topo_sorted.index(node)]
        visited[node] = True
        parents = node.parents
        stack.extend(parents)
        topo_sorted.append(node)
    return topo_sorted

def finite_difference(f, x, h=1e-4):
    """
    Calculate the finite difference
    approximation to f'(x).
    """
    return (f(x + h) - f(x)) / h

def centered_difference(f, x, h=1e-4):
    """
    Calculate the centered difference
    approximation to f'(x).
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def substitution_approximation(f, x, u, v):
    """
    Calculate the substitution approximation
    to the function f'(x).
    """
    g1 = u.T @ f(v * (x + h))
    g2 = u.T @ f(v * (x - h))
    g = (g1 - g2) / h
    return g

def grad_check(f, x, h=1e-4, eps=1e-3, fn=centered_difference):
    """
    Wrapper which automates grad checking
    a function `f` at the point `x` from
    start to finish.
    """
    assert isinstance(x, ten.Tensor)
    y = f(x)
    y.backward()
    g2 = x.grad
    approx = fn(f, x, h=h)
    return abs(g2 - approx.data) < eps
