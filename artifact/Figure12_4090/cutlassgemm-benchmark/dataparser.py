import os
import re
import csv
from matmul_data_config import _gemm_sizes, square_sizes, rectangle_sizes, roller_sizes, irregular_sizes, llm_sizes


def parse_runtime(log_path, A_layout='col', B_layout='row'):
    fp32_cudacore_runtime = []
    fp32_tensorcore_runtime = []
    fp16_cudacore_runtime = []
    fp16_tensorcore_runtime = []
    s8_cudacore_runtime = []
    s8_tensorcore_runtime = []
    with open(log_path) as f:
        render = csv.reader(f)
        # get header
        header_row = next(render)

        A_idx = header_row.index('A')
        B_idx = header_row.index('B')
        Op_class_idx = header_row.index('op_class')
        runtime_idx = header_row.index('Runtime')
        for row in render:
            if A_layout not in row[A_idx]:  # Skip rows that don't match the layout
                continue
            if B_layout not in row[B_idx]:  # Skip rows that don't match the layout
                continue
            runtime_data = float(row[runtime_idx])
            if 'cf32' in row[A_idx]:
                continue
            if 'bf32' in row[A_idx]:
                continue
            if 'f32' in row[A_idx]:
                fp32_tensorcore_runtime.append(runtime_data)
                if 'tensor' in row[Op_class_idx]:
                    fp32_tensorcore_runtime.append(runtime_data)
                else:
                    fp32_cudacore_runtime.append(runtime_data)
            elif 'f16' in row[A_idx]:
                if 'tensor' in row[Op_class_idx]:
                    fp16_tensorcore_runtime.append(runtime_data)
                else:
                    fp16_cudacore_runtime.append(runtime_data)
            elif 's8' in row[A_idx]:
                if 'tensor' in row[Op_class_idx]:
                    s8_tensorcore_runtime.append(runtime_data)
                else:
                    s8_cudacore_runtime.append(runtime_data)

    # print(fp32_tensorcore_runtime)
    min_fp32_cudacore_runtime, min_fp32_tensorcore_runtime, min_fp16_cudacore_runtime, min_fp16_tensorcore_runtime, min_s8_cudacore_runtime, min_s8_tensorcore_runtime = [min(runtime_array) if len(
        runtime_array) else 'not support' for runtime_array in (fp32_cudacore_runtime, fp32_tensorcore_runtime, fp16_cudacore_runtime, fp16_tensorcore_runtime, s8_cudacore_runtime, s8_tensorcore_runtime)]

    return min_fp32_cudacore_runtime, min_fp32_tensorcore_runtime, min_fp16_cudacore_runtime, min_fp16_tensorcore_runtime, min_s8_cudacore_runtime, min_s8_tensorcore_runtime


def get_and_print(log_path, log_case, A_layout='col', B_layout='row'):
    min_fp32_cudacore_runtime, min_fp32_tensorcore_runtime, min_fp16_cudacore_runtime, min_fp16_tensorcore_runtime, min_s8_cudacore_runtime, min_s8_tensorcore_runtime = parse_runtime(
        log_path, A_layout, B_layout)
    if log_case == 'min_fp32_cudacore_runtime':
        print(min_fp32_cudacore_runtime)
    elif log_case == 'min_fp32_tensorcore_runtime':
        print(min_fp32_tensorcore_runtime)
    elif log_case == 'min_fp16_cudacore_runtime':
        print(min_fp16_cudacore_runtime)
    elif log_case == 'min_fp16_tensorcore_runtime':
        print(min_fp16_tensorcore_runtime)
    elif log_case == 'min_s8_cudacore_runtime':
        print(min_s8_cudacore_runtime)
    elif log_case == 'min_s8_tensorcore_runtime':
        print(min_s8_tensorcore_runtime)


if __name__ == '__main__':
    # log_case = 'min_fp32_cudacore_runtime'
    # log_case = 'min_fp32_tensorcore_runtime'
    # log_case = 'min_fp16_cudacore_runtime'
    log_case = 'min_fp16_tensorcore_runtime'
    # log_case = 'min_s8_cudacore_runtime'
    # log_case = 'min_s8_tensorcore_runtime'
    log_dir = './csv_logs'
    print("llm_sizes")
    for m, n, k in llm_sizes:
        log_name = 'cutlass_shape_{0}_{1}_{2}_performance.gemm.csv'.format(
            m, k, n)
        log_path = os.path.join(log_dir, log_name)
        get_and_print(log_path, log_case, 'row', 'row')
   