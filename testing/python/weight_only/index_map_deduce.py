import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.runtime import const
from tvm.tir import IndexMap, IntImm, floordiv, floormod
from tvm import tir
index_map = IndexMap.from_func(lambda i: [i // 4, i % 4], index_dtype="int32")
initial_i = index_map.initial_indices[0]

# but what we have is i <=> i // 4
# should do inverse

block_iter_map = IndexMap.from_func(lambda i: [i // 4], index_dtype="int32")
inverse_block_iter_map = index_map.inverse([32,])

new_final_indices = index_map.map_indices([initial_i * 4])

# # tir.IndexMap([initial_i // 4], final_indices, None)
# print(new_final_indices)
