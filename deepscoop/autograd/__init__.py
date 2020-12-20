from __future__ import absolute_import

from .tensor import Tensor
from .grad_defs import define_grad

__all__ = ['Variable', 'Tensor', 'define_grad']
