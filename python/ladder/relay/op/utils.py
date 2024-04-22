# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def compute_matmul_shape(a_shape, b_shape, transpose_a, transpose_b):
    a_shape = [int(x) for x in a_shape]
    b_shape = [int(x) for x in b_shape]
    rankdiff = len(a_shape) - len(b_shape)
    if rankdiff > 0:
        b_shape = [1] * rankdiff + b_shape
    elif rankdiff < 0:
        a_shape = [1] * -rankdiff + a_shape
    out_shape = []
    for ax, bx in zip(a_shape[:-2], b_shape[:-2]):
        assert ax == bx or ax == 1 or bx == 1
        out_shape.append(max(ax, bx))
    m_value = a_shape[-1] if transpose_a else a_shape[-2]
    n_value = b_shape[-2] if transpose_b else b_shape[-1]
    out_shape += [m_value, n_value]
    return out_shape
