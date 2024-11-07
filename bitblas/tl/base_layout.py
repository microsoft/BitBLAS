# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def make_shared_to_local_linear_layout_2d(i, j, stride=16, local_size=4):

    def shared_to_local_linear_layout_2d(i, j):
        thread_id = j + (i // local_size) * stride
        local = (i % local_size)
        return thread_id, local

    return shared_to_local_linear_layout_2d
