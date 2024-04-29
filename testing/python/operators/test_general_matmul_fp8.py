# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import bitblas
from bitblas import MatmulConfig, Matmul
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


@pytest.mark.parametrize(
    "M,N,K,A_dtype,W_dtype,accum_dtype,out_dtype,layout,with_bias,group_size,with_scaling,with_zeros,zeros_mode",
    [
        (1, 1024, 1024, "e4m3_float8", "e4m3_float8", "float32", "float32", "nt", None, None, None, None,
         None),
        (1024, 1024, 1024, "e4m3_float8", "e4m3_float8", "float32", "float32", "nt", None, None, None, None,
         None),
        (1, 1024, 1024, "e5m2_float8", "e5m2_float8", "float32", "float32", "nt", None, None, None, None,
         None),
        (1024, 1024, 1024, "e5m2_float8", "e5m2_float8", "float32", "float32", "nt", None, None, None, None,
         None),
    ],
)
def test_matmul_torch_forward(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout, with_bias,
                              group_size, with_scaling, with_zeros, zeros_mode):
    import torch
    torch.random.manual_seed(0)

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=True)

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    
    def map_torch_type(intype):
        
        typemap = {
            'e4m3_float8': torch.float8_e4m3fn,
            'e5m2_float8': torch.float8_e5m2,
        }
        if intype in typemap:
            return typemap[intype]
        else:
            return getattr(torch, intype)

    numpytype_a = map_torch_type(A_dtype)
    numpytype_b = map_torch_type(W_dtype)
    numpytype_c = map_torch_type(out_dtype)
    
    torch_a = torch.rand(M*K).uniform_(-5, 5).reshape(input_shape).type(numpytype_a).cuda()
    torch_b = torch.rand(N*K).uniform_(-5, 5).reshape(weight_shape).type(numpytype_b).cuda()
    ref_out = torch.matmul(torch_a.to(torch.float32), torch_b.t().to(torch.float32)) if layout == "nt" else torch.matmul(torch_a.to(torch.float32), torch_b.to(torch.float32))
    ref_out = ref_out.to(numpytype_c)
    
    print("torch_ref_out", ref_out)
    new_torch_b = matmul.transform_weight(torch_b)
    bitblas_out = matmul(torch_a, new_torch_b)
    print("bitblas_out", bitblas_out)

# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
