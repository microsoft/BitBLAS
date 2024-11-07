# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from tvm.runtime import convert


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = (j * 16 + i)
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = (i * 16 + j)
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = (i % 4)
    return thread_id, local


def shared_16x16_to_ldmatrix_64x4_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_local_64x4_layout_C(i, j)
    return convert([thread_id, local_id])


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = (j % 4)
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = (i % 4)
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_C_smooth(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 4
    return j, i

def thread_id_shared_access_64x4_to_16x16_layout_C(thread_id, local_id):
    # This is a hacky implementation to simulate the performance
    is_smooth = os.environ.get("TILE_LANG_SMOOTH_LAYOUT") == "1"
    print(is_smooth)
    if is_smooth:
        return thread_id_shared_access_64x4_to_16x16_layout_C_smooth(thread_id, local_id)

    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j

