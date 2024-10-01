import pytest
import os
import torch
import bitblas
import threading
from bitblas import Matmul, MatmulConfig
from bitblas.cache import global_operator_cache
from bitblas import tvm as tvm
from tvm.contrib import utils

target = bitblas.utils.auto_detect_nvidia_target()
bitblas.set_log_level("DEBUG")


def get_codegen_result(ops, target):
    code = ops.get_source(target=target)
    return code


def tune_op_in_thread(thread_id, matmul_config, database_path):
    """Each thread tunes the given Matmul operation and tries to save it into the global cache."""
    matmul = Matmul(
        config=matmul_config,
        target=target,
        enable_tuning=False,
    )
    print(f"Thread {thread_id}: Starting hardware-aware tuning...")
    # matmul.hardware_aware_finetune(topk=20)
    success = False
    try:
        print(f"Thread {thread_id}: Adding operation to global cache...")
        global_operator_cache.add(matmul.config, matmul)

        global_operator_cache.save_into_database(database_path, target=target)
        assert os.path.exists(database_path), "Database file was not created."
        global_operator_cache.clear()
        assert global_operator_cache.size() == 0, "Global cache was not cleared properly."
        global_operator_cache.load_from_database(database_path, target=target)
        assert global_operator_cache.size() > 0, (
            f"Thread {thread_id}: Global cache was not loaded properly as it is empty.")

        success = True
    except Exception as hash_error:
        print(f"Thread {thread_id}: Error encountered - {hash_error}")
    assert success, f"Thread {thread_id}: Failed to add operation to global cache."


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", False, False, False, "nt"),
    ],
)
def test_global_cache_save_to_database_multithreaded(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):
    num_threads = 4
    global_operator_cache.clear()

    # For real world scenarios, all workers should share the same database path
    tempdir = utils.tempdir()
    database_path = str(tempdir.path)

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        A_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )

    # Launch four threads, each tuning the same operation
    threads = []
    for thread_id in range(num_threads):
        thread = threading.Thread(
            target=tune_op_in_thread, args=(thread_id, matmul_config, database_path))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    matmul = global_operator_cache.get(matmul_config)
    assert matmul is not None, "Matmul operation not found in cache after reload."

    # Verify that the operation produces correct results
    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)

    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda())
    inputs.append(torch.rand(weight_shape, dtype=torch.float16).cuda())
    ref_result = torch.matmul(inputs[0], inputs[1].t() if layout == "nt" else inputs[1])

    permuted_inputs = []
    if matmul.input_transform is not None:
        permuted_inputs.append(matmul.input_transform(inputs[0].cpu()).cuda())
    else:
        permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(matmul.weight_transform(inputs[1].cpu()).cuda())
    else:
        permuted_inputs.append(inputs[1])

    bitblas_output = matmul(*permuted_inputs)
    torch.testing.assert_close(bitblas_output, ref_result, rtol=1e-2, atol=1e-2)


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
