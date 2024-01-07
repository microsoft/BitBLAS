"""
This file contains test cases for the ladder policy module. It tests the execution of various operations using the ladder architecture and code generator.

The test cases include:
- Convolution operation
- Matrix multiplication operation
- ReLU activation operation
- Row reduction operation
- Depthwise convolution operation
- Average pooling operation

Each test case executes the operation using the ladder architecture and code generator, and compares the output with the reference output to ensure correctness.

To run the tests, execute the `ladder.testing.main()` function at the end of the file.
"""

import pytest
import numpy as np
import ladder
from ops import *
from ladder.graph import IRNode, OutputNode
from ladder.policy import DefaultPolicy
from ladder.reference import get_subgraph_reference_outputs


# Fixture for ladder architecture setup
@pytest.fixture(scope="module")
def arch():
    return ladder.arch.cuda()


# Fixture for the code generator
@pytest.fixture(scope="module")
def code_generator():
    return ladder.CodeGenerator()


def operation_execution(arch, code_generator, test_case):
    name, func, args = test_case
    ir, input_dict = func(*args)
    expr = "- einstein_v2('{}', {})".format(ir, str(input_dict))
    A = IRNode([None for _ in input_dict], expr)
    output_nodes = [OutputNode(A)]
    policy = DefaultPolicy(output_nodes, arch)
    configs = policy.emit_config(10)

    compile_results = []
    for config in configs:
        cpresult = code_generator.compile(
            output_nodes, config, "cuda", kernel_name="Fused"
        )
        compile_results.append(cpresult)
    ladder.utils.compile_and_load_parallel(compile_results, arch)

    best_latency = 10000
    best = None
    values = []
    for cpresult in compile_results:
        print(cpresult.config)
        if cpresult.lib is None:
            latency = 10000
        else:
            latency = cpresult.profile()
        values.append(latency)
        if latency < best_latency:
            best_latency = latency
            best = cpresult

    out = best.get_example_outputs()
    ref_out = get_subgraph_reference_outputs(output_nodes)
    for a, b in zip(out, ref_out):
        diff = np.max(np.abs(a - b))
    assert diff < 1e-2


c_lists = [
    ("C0", conv_nchw, [128, 128, 28, 28, 128, 3, 3, 1, 1, 1]),
    ("C1", conv_nchw, [128, 128, 28, 28, 128, 3, 3, 2, 1, 0]),
]


@pytest.mark.parametrize("test_case", c_lists, ids=[item[0] for item in c_lists])
def test_conv2d(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


m_lists = [
    ("M0", matmul_nn, [65536, 2, 1024]),
    ("M1", matmul_nn, [128, 4032, 1000]),
]


@pytest.mark.parametrize("test_case", m_lists, ids=[item[0] for item in m_lists])
def test_matmul(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


e_lists = [
    ("E0", relu, [227598336]),
    ("E1", relu, [6422528]),
]


@pytest.mark.parametrize("test_case", e_lists, ids=[item[0] for item in e_lists])
def test_relu(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


r_lists = [
    ("R0", row_reduce, [65536, 1024]),
    ("R1", row_reduce, [65536, 1024]),
]


@pytest.mark.parametrize("test_case", r_lists, ids=[item[0] for item in r_lists])
def test_row_reduce(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


d_lists = [
    ("D0", dwconv_nchw, [128, 84, 42, 42, 5, 5, 2, 1, 2]),
    ("D1", dwconv_nchw, [128, 42, 83, 83, 5, 5, 1, 1, 2]),
]


@pytest.mark.parametrize("test_case", d_lists, ids=[item[0] for item in d_lists])
def test_dwconv(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


p_lists = [
    ("P0", average_pooling, [21504, 42, 42, 1, 1, 2, 0]),
    ("P1", average_pooling, [86016, 11, 11, 3, 3, 2, 1]),
]


@pytest.mark.parametrize("test_case", p_lists, ids=[item[0] for item in p_lists])
def test_avgpool(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


# Run the tests
if __name__ == "__main__":
    ladder.testing.main()
