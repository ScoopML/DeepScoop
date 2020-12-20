from .optimizer import Optimizer

import logging
import deepscoop.autograd.tensor_library as tl

class RProp(Optimizer):
    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        if lr < 0:
            raise ValueError("Learning rate must be greater than 0.")
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f"Invalid etas: {etas[0]}, {etas[1]}. \
                               The first must be in 0 < eta < 1 and the second must be greater than 1.")
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super(RProp, self).__init__(params, defaults)

    def __name__(self):
        return "RProp"

    def __setstate__(self, state):
        super(RProp, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if not p.grad:
                    continue
                # Get the current param gradient
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    # Initialize state
                    state['step'] = 0
                    state['prev'] = tl.zeros(p.shape)
                    state['step_size'] = tl.full(grad.shape, group['lr'])

                eta_minus, eta_plus = group['etas']
                eta_min, eta_max = group['step_sizes']
                step_size = state['step_size']

                # Increment the current step
                state['step'] += 1

                sign = tl.sign(tl.multiply(grad, state['prev']))
                sign[sign < 0] = eta_minus
                sign[sign > 0] = eta_plus
                sign[sign == 0] = 1

                # Clip the step sizes at the min/max values
                clipped = tl.multiply(sign, step_size)
                clipped[clipped > eta_max] = eta_max
                clipped[clipped < eta_min] = eta_min

                # Create a copy of the gradient
                grad = tl.copy(grad)
                grad[sign == eta_minus] = 0

                p -= step_size * tl.sign(grad)

                state['prev'] = grad

        return loss
