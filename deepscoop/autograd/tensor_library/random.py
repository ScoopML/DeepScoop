from __future__ import absolute_import
import numpy.random as _npr
from .library import wrap_namespace

wrap_namespace(_npr.__dict__, globals())
