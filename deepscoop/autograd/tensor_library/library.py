import flamethrower.autograd.call as call
import numpy as _np

notrace_functions = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type
]

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    for name, obj in old.items():
        if obj in notrace_functions:
            new[name] = call.Primitive2(obj)
        elif callable(obj) and type(obj) is not type:
            new[name] = call.Primitive2(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

# Wrap numpy
wrap_namespace(_np.__dict__, globals())
