import math
import torch
from torch import nn
import bitblas
from bitblas import Matmul, MatmulConfig

from bitblas.cache import global_operator_cache, get_database_path
from bitblas import Matmul, MatmulConfig
from bitblas.quantization.utils import general_compress
from bitblas import auto_detect_nvidia_target
from logging import getLogger
logger = getLogger(__name__)
bitblas.set_log_level("INFO")
BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = get_database_path()



def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s =  1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1) / s
    return result.type(dtype)


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp) / s
    return result.type(dtype)   

## BitBLAS BitLinear
class BitLinear(nn.Linear):

    def __init__(self,
            *kargs,
            weight_bits=1,
            input_bits=8,
            **kwargs
        ):
        super(BitLinear, self).__init__(*kargs, **kwargs)
        """
        RMSNorm is placed outside BitLinear
        """
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        matmul_config = MatmulConfig(
            N=self.out_features,  # N dimension
            K=self.in_features,  # K dimension
            A_dtype="int8",  # activation A dtype
            W_dtype="int2",  # weight W dtype
            accum_dtype="int32",  # accumulation dtype
            out_dtype="float32",  # output dtype
            layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
            with_bias=False,  # bias
            # configs for weight only quantization
            group_size=None,  # setting for grouped quantization
            with_scaling=False,  # setting for scaling factor
            with_zeros=False,  # setting for zeros
            zeros_mode=None,  # setting for how to calculating zeros
        )
        ENABLE_TUNING = True
        self.bitblas_matmul = self._get_or_create_bitblas_operator(matmul_config, ENABLE_TUNING)
        
        self.Qp = 2 ** (self.input_bits - 1) - 1

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
            logger.info(f"Loaded {global_operator_cache.size()} operators from database.")

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            # should disable tuning for the first time because we may require loading bitblas operator from database.
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                print("BitBLAS Tuning done, appended operator to global_operator_cache.")
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def post_process_weights(self):
        sw =  1 / self.weight.abs().mean().clamp(min=1e-5)
        self.sw = sw
        quant_weight = self.weight_quant(self.weight).detach()
        quant_weight = self.bitblas_matmul.transform_weight(quant_weight)
        self.weight = nn.Parameter(quant_weight, requires_grad=False)

    def weight_quant(self, weight):
        weight = weight.float()
        s =  1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1)
        return result.type(torch.int8)

    
    def activation_quant(self, x, num_bits=8):
        x = x.float()
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp)
        return result.type(torch.int8)   

    # for the correctness evaluation.
    def native_forward(self, input):
        quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
        quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def forward_fp32_simulated(self, input):
        print("input: ", input)
        quant_input = self.activation_quant(input, self.input_bits).detach()
        quant_weight = self.weight_quant(self.weight).detach()

        fp32_simulated_input = quant_input.float()
        fp32_simulated_weight = quant_weight.float()
        fp32_simulated_out = nn.functional.linear(fp32_simulated_input, fp32_simulated_weight)
        
        sw =  1 / self.weight.abs().mean().clamp(min=1e-5)
        Qp = 2 ** (self.input_bits - 1) - 1
        si = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        # if / (si * sw) it will inf in some cases
        out = (fp32_simulated_out / si)
        out = out / sw
        out = out.half()
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def forward(self, input):
        quant_input = self.activation_quant(input, self.input_bits).detach()
        fp32_out = self.bitblas_matmul(quant_input, self.weight)
        sw = self.sw
        Qp = self.Qp
        si = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        # if / (si * sw) it will inf in some cases
        out = (fp32_out / si)
        out = out / sw
        out = out.half()
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out
    
# # Naive BitLinear
# class BitLinear(nn.Linear):

#     def __init__(self,
#             *kargs,
#             weight_bits=1,
#             input_bits=8,
#             **kwargs
#         ):
#         super(BitLinear, self).__init__(*kargs, **kwargs)
#         """
#         RMSNorm is placed outside BitLinear
#         """
#         self.weight_bits = weight_bits
#         self.input_bits = input_bits

#     def forward(self, input):
        
#         quant_input = input + (activation_quant(input, self.input_bits) - input).detach()
#         quant_weight = self.weight + (weight_quant(self.weight, self.weight_bits) - self.weight).detach()

#         out = nn.functional.linear(quant_input, quant_weight)
#         if not self.bias is None:
#             out += self.bias.view(1, -1).expand_as(out)

#         return out