from .schedule import schedule
from .ladder_intrin import C_shared_16x16_to_ldmatrix_32x8_layout
from tvm._ffi import register_func
from tvm.runtime import convert
lift = convert

@register_func("tir.index_map.shared_16x16_to_ldmatrix_32x8_layout", override=True)
def index_map_shared_16x16_to_ldmatrix_32x8_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = C_shared_16x16_to_ldmatrix_32x8_layout(i, j)
    return convert([thread_id, local_id])
