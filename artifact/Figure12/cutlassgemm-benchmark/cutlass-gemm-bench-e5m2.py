# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
from matmul_data_config import square_sizes, llm_sizes, operators_shape

if __name__ == '__main__':
    warm_up = 2
    iteration = 5
    log_dir = './csv_logs'
    for m, n, k in operators_shape:
        log_name = 'cutlass_shape_{0}_{1}_{2}_e5m2_performance.csv'.format(m, k, n)
        log_path = os.path.join(log_dir, log_name)
        command = 'cutlass_profiler  --operation=Gemm ' \
                + '--m={0} '.format(m) \
                + '--k={0} '.format(k) \
                + '--n={0} '.format(n) \
                + '--op_class=tensorop ' \
                + '--accum=f32 ' \
                + '--A=fe5m2:row '\
                + '--B=fe5m2:col '\
                + '--warmup-iterations={0} --profiling-iterations={1} '.format(warm_up, iteration) \
                + '--providers=cutlass --output={0} '.format(log_path)

        print("Currently Processing : " + command)
        _command_ret = os.system(command)
        if _command_ret :
             logging.error(command + ' execute failed, return is ', _command_ret)
        
        