import pytest
import numpy as np
import ladder
from ops import *
from ladder.graph import IRNode, OutputNode
from ladder.policy import DefaultPolicy, TCPolicy
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
    m, n = A.get_shape()
    if (
        m % 8 == 0
        and n % 8 == 0
        and (n * m) % 256 == 0
        and list(A.raxis.values())[0] % 16 == 0
    ):
        A.add_tag("tensorCoreConfig", (0, 1))
        policy = TCPolicy(output_nodes, arch)
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
    np.testing.assert_allclose(out, ref_out, rtol=1e-1, atol=1e0)


c_lists = [
    ("C0", conv_nhwc_implicit_gemm, [32, 32, 28, 28, 32, 3, 3, 1, 1, 1]),
    ("C1", conv_nhwc_implicit_gemm, [32, 32, 28, 28, 32, 3, 3, 2, 1, 0]),
]


@pytest.mark.parametrize("test_case", c_lists, ids=[item[0] for item in c_lists])
def test_conv2d(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


gemm_lists = [
    ("G0", matmul_nt, [1024, 1024, 1024, "float16"]),
    ("G1", matmul_nt, [256, 1024, 512, "float16"]),
]


@pytest.mark.parametrize("test_case", gemm_lists, ids=[item[0] for item in gemm_lists])
def test_gemm(arch, code_generator, test_case):
    operation_execution(arch, code_generator, test_case)


# Run the tests
if __name__ == "__main__":
    ladder.testing.main()
