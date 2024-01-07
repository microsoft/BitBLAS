experimental_mma_pad = False

def check_tensor_core_valid_shape(M, N, K):
    if experimental_mma_pad:
        return True
    return K % 16 == 0 and M % 8 == 0 and N % 8 == 0 and M * N % 256 == 0
