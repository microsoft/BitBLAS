from .schedule import schedule
from tvm._ffi import register_func
from tvm.runtime import convert
lift = convert


def C_shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)

@register_func("tir.index_map.shared_16x16_to_ldmatrix_32x8_layout", override=True)
def index_map_shared_16x16_to_ldmatrix_32x8_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = C_shared_16x16_to_ldmatrix_32x8_layout(i, j)
    return convert([thread_id, local_id])
