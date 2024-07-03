# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import re


def match_global_kernel(source: str) -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    matched = re.findall(pattern, source)
    assert len(matched) > 1  # may have statement before kernel
    return source.index(matched[0])


def tensor_replace_dp4a(source: str) -> str:
    # as under block reduction in tir dsl, the dp4a tensorize will fail, so we should do dp4a in post processor.
    # TODO(lei): this is a stuff that should be fixed in the tvm in the future
    pattern = r"""for\s*\(int\s*(?P<k_var>\w+)\s*=\s*0;\s*\1\s*<\s*4;\s*\+\+\1\)\s*\{\s*(?P<c_var>\w+)\[0\]\s*=\s*\(\2\[0\]\s*\+\s*\(\(\(int\)(?P<a_var>\w+)\[\(\((?P<idx_a_var>\w+)\s*\*\s*4\)\s*\+\s*\1\)\]\)\s*\*\s*\(\(int\)(?P<b_var>\w+)\[\(\((?P<idx_b_var>\w+)\s*\*\s*4\)\s*\+\s*\1\)\]\)\)\);\s*\}"""
    replacement = (r"""\2[0] = __dp4a(*(int *)&\3[((\4 * 4))],*(int *)&\5[((\6 * 4))], \2[0]);""")
    source = re.sub(pattern, replacement, source)
    return source


def tensor_remove_make_int4(source: str) -> str:
    # remove make_int4 with 16 signed char arguments
    # TODO(lei): this is a stuff that should be fixed in the tvm in the future
    source = source.replace(
        "make_int4((signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0)",
        "make_int4(0, 0, 0, 0)",
    )
    return source

def tensor_remove_make_int2(source: str) -> str:
    # remove make_int4 with 16 signed char arguments
    # TODO(lei): this is a stuff that should be fixed in the tvm in the future
    source = source.replace(
        "make_int2((signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0, (signed char)0)",
        "make_int2(0, 0)",
    )
    return source
