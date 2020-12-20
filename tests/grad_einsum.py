import numpy as np
from future.utils import string_types
from numpy.core.einsumfunc import _parse_einsum_input

def parse_einsum_input(*args):
    return _parse_einsum_input(args)

def metadata(A):
    return np.shape(A), np.ndim(A), np.result_type(A), np.iscomplexobj(A)

def grad_einsum(argnum, ans, operands_, kwargs):
    result_meta = metadata(operands_[argnum])
    def vjp(g):
        operands = operands_
        if isinstance(operands[0], string_types):  # using "ijk" convention.
            in_subs, out_subs, _ = parse_einsum_input(*operands)
            string, operands = operands[0], operands[1:]

            in_subs_list = in_subs.split(',')
            op_num = argnum - 1
            subs_wrt = in_subs_list[op_num]
            print("subs_wrt: {}".format(subs_wrt))
            rest_of_ops = operands[:op_num] + operands[op_num+1:]
            print("rest_of_ops: {}".format(rest_of_ops))
            rest_of_subs = in_subs_list[:op_num] + in_subs_list[op_num+1:]
            print("rest_of_subs: {}".format(rest_of_subs))

            # subscripts that only appear in subs_wrt (and not in other subscript lists
            # or in the output) are implicitly being summed out, as if contracted
            # against a tensor of ones. we make that tensor of ones explicit to handle
            # the necessary vjp broadcasting inside einsum.
            other_named_subs = set(''.join([out_subs] + rest_of_subs))
            naked_summed = [(i, sub) for i, sub in enumerate(subs_wrt)
                            if sub not in other_named_subs]
            print("naked_summed: {}".format(naked_summed))
            if naked_summed:
                naked_summed_dims, ones_subs = zip(*naked_summed)
                ones_subs = ''.join(ones_subs)
                ones = np.ones(np.array(operands[op_num].shape)[list(naked_summed_dims)])
                new_input_subs = ','.join([out_subs, ones_subs] + rest_of_subs)
                new_operands = (g, ones) + rest_of_ops
            else:
                new_input_subs = ','.join([out_subs] + rest_of_subs)
                print("new_input_subs: {}".format(new_input_subs))
                new_operands = (g,) + rest_of_ops

            new_subscripts = new_input_subs + '->' + subs_wrt
            print(new_subscripts, new_operands, result_meta)
            return unbroadcast(np.einsum(new_subscripts, *new_operands), result_meta)
        else:  # using (op0, sublist0, op1, sublist1, ..., sublistout) convention
            if len(operands) % 2 == 0:
                raise NotImplementedError("Need sublistout argument")
            operands = list(operands)
            rest_of_ops = [operands[-1]] + operands[:argnum] + \
                    operands[(argnum+2):-1] + [operands[argnum+1]]
            return unbroadcast_einsum(np.einsum(g, *rest_of_ops), result_meta, operands[argnum + 1])
    return vjp

def unbroadcast_einsum(x, target_meta, subscript):
    if Ellipsis not in subscript:
        return x
    elif subscript[0] == Ellipsis:
        return unbroadcast(x, target_meta, 0)
    elif subscript[-1] == Ellipsis:
        return unbroadcast(x, target_meta, -1)
    else:
        return unbroadcast(x, target_meta, subscript.index(Ellipsis))

def unbroadcast(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    while np.ndim(x) > target_ndim:
        x = np.sum(x, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = np.sum(x, axis=axis, keepdims=True)
    if np.iscomplexobj(x) and not target_iscomplex:
        x = np.real(x)
    return x

if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([[4, 5, 6], [7, 8, 9]])
    einsum_str = "ik,kj->ij"
    C = np.einsum(einsum_str, A, B)
    fn = grad_einsum(1, C, (einsum_str, A, B), {})
